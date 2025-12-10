# evaluate_ngcf.py

import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from src.data_utils.dataloader import TxtCFDataLoader
from src.data_utils.graph_builder import GraphBuilder
from src.models.NGCF import NGCF
from src.data_utils.graph_builder_time_decay import TimeDecayGraphBuilder

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_ground_truth(user_items: Dict[int, List[int]]):
    users = sorted(user_items.keys())
    gt_dict = {u: set(items) for u, items in user_items.items()}
    return users, gt_dict


def metrics_at_k(topk_idx: List[int], gt_items: set, k: int):
    """Tính Precision, Recall, HitRate, NDCG, AP cho 1 user."""
    if len(gt_items) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    hits = [1 if i in gt_items else 0 for i in topk_idx]
    n_rel = len(gt_items)

    # HitRate@K
    hit_any = 1.0 if any(hits) else 0.0

    # Precision@K, Recall@K
    n_hit = sum(hits)
    precision = n_hit / len(topk_idx)
    recall = n_hit / n_rel

    # NDCG@K
    dcg = 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            dcg += 1.0 / np.log2(rank + 1)
    idcg = 0.0
    max_rank = min(n_rel, len(topk_idx))
    for rank in range(1, max_rank + 1):
        idcg += 1.0 / np.log2(rank + 1)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # AP@K
    ap = 0.0
    hit_cnt = 0
    for rank, h in enumerate(hits, start=1):
        if h:
            hit_cnt += 1
            ap += hit_cnt / rank
    ap /= min(n_rel, len(topk_idx))

    return precision, recall, hit_any, ndcg, ap


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NGCF (H&M)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="e.g. data/h_m")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained NGCF .pt")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use_time_decay",
        action="store_true",
        help="Use time-decay weighted graph (must match training)"
    )
    parser.add_argument(
        "--time_weight_csv",
        type=str,
        default=None,
        help="Path to train_time_weights.csv; default <data_dir>/train_time_weights.csv"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, fallback to CPU.")
        device = "cpu"
    print("Using device:", device)

    # 1) Load data
    loader = TxtCFDataLoader(args.data_dir, verbose=True)

    # Chọn val/test ground truth
    if args.split == "val":
        user_items_eval = loader.val
    else:
        user_items_eval = loader.test

    users_eval, gt_dict = build_ground_truth(user_items_eval)
    num_eval_users = len(users_eval)
    print(f"\n[Eval] Split: {args.split}, Users: {num_eval_users}")

    # 2) Build graph TRAIN-ONLY cho NGCF (có self-loop A+I)
    train_pos = loader.train

    if args.time_weight_csv is None:
        args.time_weight_csv = os.path.join(args.data_dir, "train_time_weights.csv")

    if args.use_time_decay:
        print("[Graph] Using TIME-DECAY NGCF graph (A + I) for eval...")
        gb = TimeDecayGraphBuilder(
            num_users=loader.num_users,
            num_items=loader.num_items,
            weight_csv=args.time_weight_csv,
            add_self_loop=True,
            verbose=True,
        )
    else:
        print("[Graph] Using BINARY NGCF graph (A + I) for eval...")
        gb = GraphBuilder(
            num_users=loader.num_users,
            num_items=loader.num_items,
            train_user_items=train_pos,
            add_self_loop=True,
        )

    adj = gb.build_normalized_adj(device=device)

    # 3) Load checkpoint NGCF
    ckpt = torch.load(args.checkpoint, map_location=device)

    args_ckpt = ckpt["args"]  # đã là dict do vars(args)

    num_users = ckpt["num_users"]
    num_items = ckpt["num_items"]

    emb_dim = args_ckpt.get("emb_dim", 64)
    layer_sizes = args_ckpt.get("layer_sizes", [emb_dim, emb_dim])
    reg_weight = args_ckpt.get("reg_weight", 1e-4)
    mess_dropout = args_ckpt.get("mess_dropout", 0.1)
    leaky_relu_slope = args_ckpt.get("leaky_relu_slope", 0.2)

    model = NGCF(
        num_users=num_users,
        num_items=num_items,
        emb_dim=emb_dim,
        layer_sizes=layer_sizes,
        reg_weight=reg_weight,
        mess_dropout=mess_dropout,
        leaky_relu_slope=leaky_relu_slope,
    ).to(device)

    state_dict = ckpt.get("model_state")  # vì bạn lưu tên này
    model.load_state_dict(state_dict)
    model.eval()

    # 4) Propagate một lần
    with torch.no_grad():
        all_embs = model.propagate(adj)
        user_embs = all_embs[:loader.num_users]
        item_embs = all_embs[loader.num_users:]

    batch_size = args.batch_size
    K = args.K

    precisions, recalls, hitrates, ndcgs, aps = [], [], [], [], []

    with torch.no_grad():
        for start in range(0, num_eval_users, batch_size):
            end = min(start + batch_size, num_eval_users)
            batch_users = users_eval[start:end]
            batch_users_t = torch.tensor(batch_users, dtype=torch.long, device=device)

            u_emb = user_embs[batch_users_t]
            scores = torch.matmul(u_emb, item_embs.t())
            _, topk_indices = torch.topk(scores, K, dim=1)

            for i, u in enumerate(batch_users):
                gt_items = gt_dict[u]
                topk_idx = topk_indices[i].cpu().numpy().tolist()

                p, r, h, n, ap = metrics_at_k(topk_idx, gt_items, K)
                precisions.append(p)
                recalls.append(r)
                hitrates.append(h)
                ndcgs.append(n)
                aps.append(ap)

            if (start // batch_size) % 10 == 0:
                print(f"[Eval] Processed {end}/{num_eval_users} users ...")

    prec_mean = float(np.mean(precisions))
    rec_mean = float(np.mean(recalls))
    hit_mean = float(np.mean(hitrates))
    ndcg_mean = float(np.mean(ndcgs))
    map_mean = float(np.mean(aps))

    print("\n========== EVAL RESULT ==========")
    print(f"Split:      {args.split}")
    print(f"K:          {K}")
    print(f"Precision@{K}: {prec_mean:.4f}")
    print(f"Recall@{K}:    {rec_mean:.4f}")
    print(f"HitRate@{K}:   {hit_mean:.4f}")
    print(f"NDCG@{K}:      {ndcg_mean:.4f}")
    print(f"MAP@{K}:       {map_mean:.4f}")
    print("=================================")


if __name__ == "__main__":
    main()

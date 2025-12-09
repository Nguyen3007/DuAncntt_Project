# train_als.py

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src.data_utils.dataloader import TxtCFDataLoader


# =========================
# Utils
# =========================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_als_on_split(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    loader: TxtCFDataLoader,
    split: str = "val",
    K: int = 20,
    batch_users: int = 1024,
    device_str: str = "cpu",
):
    """
    Đánh giá ALS trên split 'val' hoặc 'test':
    - Không mask train items (giống LightGCN/NGCF của bạn)
    - Mỗi user có đúng 1 ground-truth item (LOO)
    - Trả về: precision, recall (=hit), ndcg, map
    """
    assert split in {"val", "test"}

    if split == "val":
        truth = loader.get_val_truth()   # {u: item_val}
    else:
        truth = loader.get_test_truth()  # {u: item_test}

    users = sorted(truth.keys())
    n_users = len(users)
    if n_users == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit": 0.0,
            "ndcg": 0.0,
            "map": 0.0,
        }

    # Đưa về torch để tận dụng GPU (nếu có)
    device = torch.device(
        "cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"[Eval] Using device: {device}")

    user_embs = torch.from_numpy(user_factors).to(device)   # [U, d]
    item_embs = torch.from_numpy(item_factors).to(device)   # [I, d]

    total_hits = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    total_map = 0.0

    import math

    for start in range(0, n_users, batch_users):
        end = min(start + batch_users, n_users)
        batch_u = users[start:end]

        u_tensor = torch.tensor(batch_u, dtype=torch.long, device=device)
        u_emb = user_embs[u_tensor]  # [B, d]

        # scores = U * I^T
        scores = torch.matmul(u_emb, item_embs.t())  # [B, n_items]
        _, topk_idx = torch.topk(scores, K, dim=1)   # [B, K]
        topk_idx = topk_idx.cpu().numpy()

        for i, u in enumerate(batch_u):
            gt = truth[u]                 # 1 ground-truth item
            rec_list = topk_idx[i].tolist()

            if gt in rec_list:
                total_hits += 1.0
                rank = rec_list.index(gt)    # 0-based
                total_recall += 1.0          # |G|=1
                total_precision += 1.0 / K
                total_ndcg += 1.0 / math.log2(rank + 2)
                total_map += 1.0 / (rank + 1)
            # else: cộng 0

        if (start // batch_users) % 20 == 0:
            print(f"[Eval] Processed {end}/{n_users} users ...")

    precision = total_precision / n_users
    recall = total_recall / n_users
    hitrate = total_hits / n_users
    ndcg = total_ndcg / n_users
    map_k = total_map / n_users

    print("\n========== ALS EVAL RESULT ==========")
    print(f"Split:      {split}")
    print(f"K:          {K}")
    print(f"Precision@{K}: {precision:.4f}")
    print(f"Recall@{K}:    {recall:.4f}")
    print(f"HitRate@{K}:   {hitrate:.4f}")
    print(f"NDCG@{K}:      {ndcg:.4f}")
    print(f"MAP@{K}:       {map_k:.4f}")
    print("=====================================\n")

    return {
        "precision": precision,
        "recall": recall,
        "hit": hitrate,
        "ndcg": ndcg,
        "map": map_k,
    }


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train ALS (implicit) on CF txt data")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/h_m")

    # ALS hyperparams
    parser.add_argument("--factors", type=int, default=64, help="Embedding dim")
    parser.add_argument("--reg", type=float, default=1e-2, help="L2 regularization")
    parser.add_argument("--iterations", type=int, default=20, help="#ALS iterations")
    parser.add_argument(
        "--alpha",
        type=float,
        default=20.0,
        help="Confidence scaling for implicit data (Cui = 1 + alpha * r_ui)",
    )

    # Eval
    parser.add_argument("--eval_K", type=int, default=20)
    parser.add_argument("--eval_batch_users", type=int, default=1024)

    # Device cho eval (train ALS chạy CPU)
    parser.add_argument("--device", type=str, default="cpu")

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_name = Path(args.data_dir).name  # "h_m" hoặc "vibrent"

    if args.model_name is None:
        args.model_name = f"als_{dataset_name}.npz"

    device_str = args.device
    print("=== ALS TRAINING CONFIG ===")
    print(f"Data dir    : {args.data_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"factors     : {args.factors}")
    print(f"reg         : {args.reg}")
    print(f"iterations  : {args.iterations}")
    print(f"alpha       : {args.alpha}")
    print(f"checkpoint  : {Path(args.checkpoint_dir) / args.model_name}")
    print("============================\n")

    # 1) Load data từ train/val/test txt
    loader = TxtCFDataLoader(args.data_dir, verbose=True)
    train_pos = loader.get_train_pos()  # {u: [i1, i2, ...]}

    num_users = loader.num_users
    num_items = loader.num_items

    # 2) Xây sparse matrix implicit: user_items (U x I)
    #    mỗi (u, i) trong train_pos có r_ui = 1
    rows = []
    cols = []
    data = []

    for u, items in train_pos.items():
        for i in items:
            rows.append(u)  # row = user index
            cols.append(i)  # col = item index
            data.append(1.0)  # implicit count

    user_items = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(num_users, num_items),
        dtype=np.float32,
    ).tocsr()

    print("\n=== Sparse Matrix (user_items) ===")
    print("shape:", user_items.shape)
    print("nnz  :", user_items.nnz)
    print("==============================\n")

    # 3) Train ALS implicit
    #    implicit.als cần item_users, và thường dùng alpha * item_users
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.reg,
        iterations=args.iterations,
        use_gpu=False,
    )

    # Confidence = 1 + alpha * r_ui, nhưng implicit ALS mặc định coi matrix là (alpha * r_ui),
    print("Fitting ALS model ...")
    confidence = user_items * args.alpha
    model.fit(confidence, show_progress=True)
    print("Done ALS training.\n")

    user_factors = model.user_factors    # [U, d]
    item_factors = model.item_factors    # [I, d]

    # 4) Eval trên VAL + TEST
    print("=== EVAL on VAL ===")
    metrics_val = eval_als_on_split(
        user_factors=user_factors,
        item_factors=item_factors,
        loader=loader,
        split="val",
        K=args.eval_K,
        batch_users=args.eval_batch_users,
        device_str=device_str,
    )

    print("=== EVAL on TEST ===")
    metrics_test = eval_als_on_split(
        user_factors=user_factors,
        item_factors=item_factors,
        loader=loader,
        split="test",
        K=args.eval_K,
        batch_users=args.eval_batch_users,
        device_str=device_str,
    )

    # 5) Lưu checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / args.model_name

    np.savez(
        ckpt_path,
        user_factors=user_factors,
        item_factors=item_factors,
        num_users=num_users,
        num_items=num_items,
        args=vars(args),
        metrics_val=metrics_val,
        metrics_test=metrics_test,
    )

    print(f"✅ Saved ALS model to: {ckpt_path}")


if __name__ == "__main__":
    main()

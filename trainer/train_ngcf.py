# train_ngcf.py

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data_utils.dataloader import TxtCFDataLoader
from src.data_utils.graph_builder import GraphBuilder
from src.models.NGCF import NGCF
from src.data_utils.graph_builder_time_decay import TimeDecayGraphBuilder

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


def sample_batch_bpr(
    train_user_items: Dict[int, List[int]],
    num_items: int,
    batch_size: int,
    device: str,
):
    """
    BPR negative sampling:
      - Random batch_size user (có thể trùng)
      - Mỗi user:
          + chọn 1 positive item từ lịch sử
          + chọn 1 negative item không nằm trong lịch sử
    """
    users = list(train_user_items.keys())
    batch_users = np.random.choice(users, size=batch_size, replace=True)

    pos_items = []
    neg_items = []

    for u in batch_users:
        pos_list = train_user_items[u]
        p = random.choice(pos_list)

        user_pos_set = set(pos_list)
        while True:
            n = random.randint(0, num_items - 1)
            if n not in user_pos_set:
                break

        pos_items.append(p)
        neg_items.append(n)

    users_t = torch.tensor(batch_users, dtype=torch.long, device=device)
    pos_t = torch.tensor(pos_items, dtype=torch.long, device=device)
    neg_t = torch.tensor(neg_items, dtype=torch.long, device=device)
    return users_t, pos_t, neg_t


# =========================
# Metrics (1 GT item / user)
# =========================

def _metrics_for_user(rec_items, gt_item, K: int):
    """
    rec_items: list item id length K (top-K)
    gt_item: ground-truth item
    """
    if gt_item in rec_items:
        rank = rec_items.index(gt_item)  # 0-based
        hit = 1.0
        precision = 1.0 / K
        recall = 1.0
        ndcg = 1.0 / np.log2(rank + 2)
        ap = 1.0 / (rank + 1)  # AP@K when only 1 positive
    else:
        hit = 0.0
        precision = 0.0
        recall = 0.0
        ndcg = 0.0
        ap = 0.0
    return precision, recall, hit, ndcg, ap


def eval_ngcf_on_split(
    model: NGCF,
    adj: torch.Tensor,
    loader: TxtCFDataLoader,
    split: str,
    device: str,
    K: int = 20,
    user_batch_size: int = 1024,
) -> Dict[str, float]:
    """
    Đánh giá NGCF trên split 'val' hoặc 'test' với:
      - Precision@K
      - Recall@K
      - HitRate@K
      - NDCG@K
      - MAP@K
    """
    model.eval()
    assert split in ["val", "test"]
    if split == "val":
        user_truth = loader.get_val_truth()
    else:
        user_truth = loader.get_test_truth()

    all_users = np.array(list(user_truth.keys()), dtype=np.int64)
    all_gt = np.array(list(user_truth.values()), dtype=np.int64)
    num_users = len(all_users)
    print(f"\n[Eval] Split: {split}, Users: {num_users}")

    with torch.no_grad():
        all_embs = model.propagate(adj)

    prec_list, rec_list, hit_list, ndcg_list, map_list = [], [], [], [], []

    for start in range(0, num_users, user_batch_size):
        end = min(start + user_batch_size, num_users)
        batch_users_np = all_users[start:end]
        batch_gt_np = all_gt[start:end]

        batch_users = torch.from_numpy(batch_users_np).long().to(device)

        with torch.no_grad():
            scores = model.full_sort_scores(all_embs, batch_users)  # (B, num_items)
            topk_items = torch.topk(scores, k=K, dim=1).indices.cpu().numpy()

        for i in range(end - start):
            rec_items = topk_items[i].tolist()
            gt_item = int(batch_gt_np[i])
            p, r, h, n, ap = _metrics_for_user(rec_items, gt_item, K)
            prec_list.append(p)
            rec_list.append(r)
            hit_list.append(h)
            ndcg_list.append(n)
            map_list.append(ap)

        if (start // user_batch_size) % 20 == 0:
            print(f"[Eval] Processed {end}/{num_users} users ...")

    precision = float(np.mean(prec_list))
    recall = float(np.mean(rec_list))
    hitrate = float(np.mean(hit_list))
    ndcg = float(np.mean(ndcg_list))
    mAP = float(np.mean(map_list))

    print("\n========== EVAL RESULT ==========")
    print(f"Split:      {split}")
    print(f"K:          {K}")
    print(f"Precision@{K}: {precision:.4f}")
    print(f"Recall@{K}:    {recall:.4f}")
    print(f"HitRate@{K}:   {hitrate:.4f}")
    print(f"NDCG@{K}:      {ndcg:.4f}")
    print(f"MAP@{K}:       {mAP:.4f}")
    print("=================================\n")

    return {
        "precision": precision,
        "recall": recall,
        "hitrate": hitrate,
        "ndcg": ndcg,
        "map": mAP,
    }


# =========================
# Train
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train NGCF on H&M subset")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/h_m")

    # Model
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--layer_sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--mess_dropout", type=float, default=0.1)
    parser.add_argument("--leaky_relu_slope", type=float, default=0.2)

    # Train
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=800,
        help="Số batch/epoch. 0 = auto dựa trên #interactions",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Eval / early stopping
    parser.add_argument("--val_K", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--eval_user_batch_size", type=int, default=2048)

    # Device & checkpoint
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--best_name", type=str, default="ngcf_hm_best.pt")
    parser.add_argument("--last_name", type=str, default="ngcf_hm_last.pt")

    # Graph / time-decay options
    parser.add_argument(
        "--use_time_decay",
        action="store_true",
        help="Use time-decay weighted graph instead of binary graph"
    )
    parser.add_argument(
        "--time_weight_csv",
        type=str,
        default=None,
        help="Path to train_time_weights.csv (u,v,weight). "
             "If None, will use <data_dir>/train_time_weights.csv"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # Nếu dùng time-decay mà vẫn để tên default thì tự đổi để phân biệt
    if args.use_time_decay and args.best_name == "ngcf_hm_best.pt":
        args.best_name = "ngcf_hm_best_td.pt"
        args.last_name = "ngcf_hm_last_td.pt"
        print("🔄 [Checkpoint] Time-decay mode → rename to:",
              args.best_name, args.last_name)
    seed_everything(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, fallback to CPU.")
        device = "cpu"
    print("Using device:", device)

    # 1) Load data
    loader = TxtCFDataLoader(args.data_dir, verbose=True)
    train_pos = loader.get_train_pos()

    num_train_interactions = sum(len(v) for v in train_pos.values())
    print(f"\n[Data] Train interactions: {num_train_interactions:,}")
    print(f"[Data] Users: {loader.num_users:,} | Items: {loader.num_items:,}")

    # 2) Build graph (NGCF: add_self_loop=True)
    # Nếu không truyền path riêng thì mặc định lấy trong data_dir
    if args.time_weight_csv is None:
        args.time_weight_csv = os.path.join(args.data_dir, "train_time_weights.csv")

    if args.use_time_decay:
        print("[Graph] Building TIME-DECAY NGCF graph (A + I)...")
        gb = TimeDecayGraphBuilder(
            num_users=loader.num_users,
            num_items=loader.num_items,
            weight_csv=args.time_weight_csv,
            add_self_loop=True,  # NGCF: A + I
            verbose=True,
        )
    else:
        print("[Graph] Building BINARY NGCF graph (A + I)...")
        gb = GraphBuilder(
            num_users=loader.num_users,
            num_items=loader.num_items,
            train_user_items=train_pos,
            add_self_loop=True,
        )

    adj = gb.build_normalized_adj(device=device)
    print("[Graph] Done.\n")

    # 3) Init model + optimizer
    model = NGCF(
        num_users=loader.num_users,
        num_items=loader.num_items,
        emb_dim=args.emb_dim,
        layer_sizes=args.layer_sizes,
        reg_weight=args.reg_weight,
        mess_dropout=args.mess_dropout,
        leaky_relu_slope=args.leaky_relu_slope,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) Train loop
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    else:
        steps_per_epoch = max(1, num_train_interactions // args.batch_size)

    print(
        f"\n[Train] steps_per_epoch = {steps_per_epoch} "
        f"(train_interactions={num_train_interactions:,}, batch_size={args.batch_size})"
    )

    best_recall = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step in range(1, steps_per_epoch + 1):
            users, pos_items, neg_items = sample_batch_bpr(
                train_user_items=train_pos,
                num_items=loader.num_items,
                batch_size=args.batch_size,
                device=device,
            )

            optimizer.zero_grad()
            loss, log_dict = model.bpr_loss_slow(
                users=users,
                pos_items=pos_items,
                neg_items=neg_items,
                adj=adj,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

            if step % 50 == 0 or step == 1:
                avg_step_loss = epoch_loss / step
                print(
                    f"Epoch {epoch:02d} | Step {step:4d}/{steps_per_epoch} "
                    f"| Loss {avg_step_loss:.4f} "
                    f"(bpr={log_dict['bpr']:.4f}, reg={log_dict['reg']:.4f})"
                )

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"\n[Epoch {epoch}] Avg train loss: {avg_epoch_loss:.4f}")

        # 5) Eval trên val
        if epoch % args.eval_every == 0:
            metrics_val = eval_ngcf_on_split(
                model=model,
                adj=adj,
                loader=loader,
                split="val",
                device=device,
                K=args.val_K,
                user_batch_size=args.eval_user_batch_size,
            )
            recall_val = metrics_val["recall"]
            print(
                f"[Epoch {epoch}] VAL@{args.val_K} | "
                f"Precision: {metrics_val['precision']:.6f} | "
                f"Recall: {metrics_val['recall']:.6f} | "
                f"HitRate: {metrics_val['hitrate']:.6f} | "
                f"NDCG: {metrics_val['ndcg']:.6f} | "
                f"MAP: {metrics_val['map']:.6f}"
            )

            # early stopping theo Recall@K
            if recall_val > best_recall:
                best_recall = recall_val
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(
                    f"__ New best model at epoch {epoch} "
                    f"with Recall@{args.val_K} = {best_recall:.6f}"
                )
            else:
                patience_counter += 1
                print(
                    f"__ No improvement. patience_counter = "
                    f"{patience_counter}/{args.early_stop_patience}"
                )

        # Save last checkpoint mỗi epoch
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        last_path = Path(args.checkpoint_dir) / args.last_name
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "best_recall": best_recall,
                "best_epoch": best_epoch,
                "num_users": loader.num_users,
                "num_items": loader.num_items,
            },
            last_path,
        )
        print(f"💾 Saved last checkpoint to: {last_path}\n")

        if patience_counter >= args.early_stop_patience:
            print(
                f"⏹ Early stopping at epoch {epoch} "
                f"(best epoch = {best_epoch}, best Recall@{args.val_K} = {best_recall:.6f})"
            )
            break

    # 6) Save best model
    if best_state is not None:
        best_path = Path(args.checkpoint_dir) / args.best_name
        torch.save(
            {
                "epoch": best_epoch,
                "model_state": best_state,
                "args": vars(args),
                "best_recall": best_recall,
                "num_users": loader.num_users,
                "num_items": loader.num_items,
            },
            best_path,
        )
        print(f"\n__ Saved BEST model to: {best_path}")
    else:
        print("\n__ No best_state saved (no eval?)")


if __name__ == "__main__":
    main()

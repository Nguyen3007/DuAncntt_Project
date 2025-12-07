# train_lightGCN_v2.py

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_utils.dataloader import TxtCFDataLoader
from src.data_utils.graph_builder import GraphBuilder
from src.models.LightGCN import LightGCN


# ============================================================
# Utils
# ============================================================

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
    device: torch.device,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    BPR negative sampling:
      - Chọn batch_size user (có thể trùng, replace=True)
      - Với mỗi user:
          + Chọn 1 positive item từ lịch sử
          + Chọn 1 negative item không nằm trong lịch sử
    """
    users = list(train_user_items.keys())
    batch_users = np.random.choice(users, size=batch_size, replace=True)

    pos_items = []
    neg_items = []

    for u in batch_users:
        pos_list = train_user_items[u]
        p = random.choice(pos_list)  # positive item

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


# ============================================================
# Evaluation (VAL / TEST) – dùng get_val_truth / get_test_truth
# ============================================================

def eval_lightgcn_on_split(
    model: LightGCN,
    adj: torch.Tensor,
    loader: TxtCFDataLoader,
    device: torch.device,
    split: str = "val",
    K: int = 20,
    batch_users: int = 1024,
):
    """
    Eval đơn giản cho early stopping + report:
    - Không mask train items (cho phép recommend lại item đã mua).
    - Mỗi user có đúng 1 ground-truth item (leave-last-1),
      lấy từ get_val_truth() hoặc get_test_truth().
    - Metrics: Precision@K, Recall@K, HitRate@K, NDCG@K, MAP@K (dạng [0,1])
    """
    assert split in {"val", "test"}

    model.eval()
    with torch.no_grad():
        all_embs = model.propagate(adj)
        user_embs = all_embs[:loader.num_users]      # [U, d]
        item_embs = all_embs[loader.num_users:]      # [I, d]

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
                rank = rec_list.index(gt)   # 0-based
                total_recall += 1.0         # |G| = 1
                total_precision += 1.0 / K
                total_ndcg += 1.0 / math.log2(rank + 2)
                total_map += 1.0 / (rank + 1)
            # else: add 0 cho mọi metric

    precision = total_precision / n_users
    recall = total_recall / n_users
    hitrate = total_hits / n_users
    ndcg = total_ndcg / n_users
    map_k = total_map / n_users

    return {
        "precision": precision,
        "recall": recall,
        "hit": hitrate,
        "ndcg": ndcg,
        "map": map_k,
    }


# ============================================================
# Main training with early stopping
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGCN with early stopping (H&M)")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/h_m")

    # Model
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--reg_weight", type=float, default=1e-3)

    # Train
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Early stopping & eval
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--val_K", type=int, default=20)
    parser.add_argument("--val_batch_users", type=int, default=1024)

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--best_name", type=str, default="lightgcn_hm_best.pt")
    parser.add_argument("--last_name", type=str, default="lightgcn_hm_last.pt")

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, fallback to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print("Using device:", device)

    # 1) Load data
    loader = TxtCFDataLoader(args.data_dir, verbose=True)
    train_pos = loader.get_train_pos()

    num_train_interactions = sum(len(v) for v in train_pos.values())
    print(f"\n[Data] Train interactions: {num_train_interactions:,}")
    print(f"[Data] Users: {loader.num_users:,} | Items: {loader.num_items:,}\n")

    # 2) Build graph
    print("[Graph] Building normalized adjacency...")
    gb = GraphBuilder(
        num_users=loader.num_users,
        num_items=loader.num_items,
        train_user_items=train_pos,
    )
    adj = gb.build_normalized_adj(device=device)
    print("[Graph] Done.\n")

    # 3) Init model + optimizer
    model = LightGCN(
        num_users=loader.num_users,
        num_items=loader.num_items,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        reg_weight=args.reg_weight,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4) Training loop with early stopping
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_recall = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state = None

    steps_per_epoch = args.steps_per_epoch

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for step in range(1, steps_per_epoch + 1):
            users, pos_items, neg_items = sample_batch_bpr(
                train_user_items=train_pos,
                num_items=loader.num_items,
                batch_size=args.batch_size,
                device=device,
            )

            loss, loss_dict = model.bpr_loss_slow(
                users=users,
                pos_items=pos_items,
                neg_items=neg_items,
                adj=adj,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if step % 50 == 0 or step == 1:
                avg_step_loss = float(np.mean(epoch_losses))
                print(
                    f"Epoch {epoch:02d} | Step {step:4d}/{steps_per_epoch} | "
                    f"Loss {avg_step_loss:.4f} "
                    f"(bpr={loss_dict['bpr']:.4f}, reg={loss_dict['reg']:.4f})"
                )

        avg_loss = float(np.mean(epoch_losses))
        print(f"\n[Epoch {epoch}] Avg train loss: {avg_loss:.4f}")

        # 5) Eval on VAL
        metrics_val = eval_lightgcn_on_split(
            model=model,
            adj=adj,
            loader=loader,
            device=device,
            split="val",
            K=args.val_K,
            batch_users=args.val_batch_users,
        )

        print(
            f"[Epoch {epoch}] VAL@{args.val_K} | "
            f"Precision: {metrics_val['precision']:.6f} | "
            f"Recall: {metrics_val['recall']:.6f} | "
            f"HitRate: {metrics_val['hit']:.6f} | "
            f"NDCG: {metrics_val['ndcg']:.6f} | "
            f"MAP: {metrics_val['map']:.6f}"
        )

        cur_recall = metrics_val["recall"]
        if cur_recall > best_recall + 1e-6:
            best_recall = cur_recall
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(
                f"⭐ New best model at epoch {epoch} "
                f"with Recall@{args.val_K} = {best_recall:.6f}"
            )
        else:
            patience_counter += 1
            print(
                f"[EarlyStop] No improvement in Recall@{args.val_K}. "
                f"patience = {patience_counter}/{args.early_stop_patience}"
            )

        # Save last checkpoint
        last_path = Path(args.checkpoint_dir) / args.last_name
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "best_recall": best_recall,
                "best_epoch": best_epoch,
            },
            last_path,
        )
        print(f"💾 Saved last checkpoint to: {last_path}\n")

        if patience_counter >= args.early_stop_patience:
            print(
                f"⏹ Early stopping triggered at epoch {epoch} "
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
            },
            best_path,
        )
        print(f"\n✅ Saved BEST model to: {best_path}")
    else:
        print("\n⚠ Training finished but no best_state was recorded.")


if __name__ == "__main__":
    main()

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


def sample_batch_bpr(
    train_user_items: Dict[int, List[int]],
    num_items: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Mini-batch BPR sampling.
    - Chọn ngẫu nhiên user có history
    - Mỗi user: 1 positive, 1 negative
    """
    users = []
    pos_items = []
    neg_items = []

    all_users = list(train_user_items.keys())
    for _ in range(batch_size):
        u = random.choice(all_users)
        items_u = train_user_items[u]
        if not items_u:
            continue
        i = random.choice(items_u)

        # sample negative j
        while True:
            j = random.randint(0, num_items - 1)
            if j not in items_u:
                break

        users.append(u)
        pos_items.append(i)
        neg_items.append(j)

    users = torch.tensor(users, dtype=torch.long, device=device)
    pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)
    neg_items = torch.tensor(neg_items, dtype=torch.long, device=device)
    return users, pos_items, neg_items


# ============================================================
# Quick evaluation on VAL (subset users) for early stopping
# ============================================================

def eval_lightgcn_on_split(
    model: LightGCN,
    adj: torch.Tensor,
    loader: TxtCFDataLoader,
    split: str = "val",
    K: int = 20,
    device: torch.device = torch.device("cpu"),
    max_eval_users: int = 50000,
    batch_users: int = 1024,
) -> Dict[str, float]:
    """
    Eval đơn giản cho early stopping.
    - Không mask train items (cho phép recommend lại item đã mua).
    - Giả định mỗi user có 1 item ground-truth trong val/test
      (leave-last-1).
    - Metrics: Precision@K, Recall@K, HitRate@K, NDCG@K, MAP@K
    """
    assert split in {"val", "test"}

    model.eval()
    with torch.no_grad():
        all_embs = model.propagate(adj)
        user_embs = all_embs[:loader.num_users]
        item_embs = all_embs[loader.num_users:]

        if split == "val":
            user_pos = loader.get_val_pos()
        else:
            user_pos = loader.get_test_pos()

        user_ids = sorted(user_pos.keys())
        if max_eval_users is not None and len(user_ids) > max_eval_users:
            # lấy subset user để eval nhanh hơn
            user_ids = user_ids[:max_eval_users]

        n_users_eval = len(user_ids)
        if n_users_eval == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "hit": 0.0,
                "ndcg": 0.0,
                "map": 0.0,
            }

        hit_sum = 0.0
        prec_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        map_sum = 0.0

        import math

        for start in range(0, n_users_eval, batch_users):
            end = min(start + batch_users, n_users_eval)
            batch_u = user_ids[start:end]

            u_tensor = torch.tensor(batch_u, dtype=torch.long, device=device)
            u_emb = user_embs[u_tensor]  # [B, d]

            # scores = U * I^T
            scores = torch.matmul(u_emb, item_embs.t())  # [B, n_items]

            topk_scores, topk_idx = torch.topk(scores, K, dim=1)  # [B, K]

            topk_idx = topk_idx.cpu().numpy()

            for i, u in enumerate(batch_u):
                gt_items = user_pos[u]
                if not gt_items:
                    continue
                # assume 1 gt item / user
                gt = gt_items[0]
                rec_list = topk_idx[i].tolist()

                if gt in rec_list:
                    hit_sum += 1.0
                    rank = rec_list.index(gt)  # 0-based
                    prec_sum += 1.0 / K
                    recall_sum += 1.0  # |G| = 1
                    ndcg_sum += 1.0 / math.log2(rank + 2)
                    map_sum += 1.0 / (rank + 1)
                # else: all metrics contribution = 0

        precision = prec_sum / n_users_eval
        recall = recall_sum / n_users_eval
        hitrate = hit_sum / n_users_eval
        ndcg = ndcg_sum / n_users_eval
        map_k = map_sum / n_users_eval

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

def main():
    parser = argparse.ArgumentParser(description="Train LightGCN with early stopping")

    parser.add_argument("--data_dir", type=str, default="data/h_m")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--reg_weight", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--val_K", type=int, default=20)
    parser.add_argument("--val_max_users", type=int, default=50000)
    parser.add_argument("--val_batch_users", type=int, default=1024)

    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--best_name", type=str, default="lightgcn_hm_best.pt")
    parser.add_argument("--last_name", type=str, default="lightgcn_hm_last.pt")

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    loader = TxtCFDataLoader(args.data_dir, verbose=True)

    print(f"\n[Data] Train interactions: {loader.num_interactions_train:,}")
    print(f"[Data] Users: {loader.num_users:,} | Items: {loader.num_items:,}\n")

    train_user_items = loader.get_train_pos()

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------
    print("[Graph] Building normalized adjacency...")
    gb = GraphBuilder(
        num_users=loader.num_users,
        num_items=loader.num_items,
        train_user_items=train_user_items,
    )
    adj = gb.build_normalized_adj(device=device)
    print("[Graph] Done.\n")

    # --------------------------------------------------------
    # Model & optimizer
    # --------------------------------------------------------
    model = LightGCN(
        num_users=loader.num_users,
        num_items=loader.num_items,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------------
    # Training loop with early stopping
    # --------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_recall = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for step in range(1, args.steps_per_epoch + 1):
            users, pos_items, neg_items = sample_batch_bpr(
                train_user_items,
                loader.num_items,
                args.batch_size,
                device,
            )

            loss, loss_dict = model.bpr_loss_slow(
                users, pos_items, neg_items, adj
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if step % 50 == 0 or step == 1:
                print(
                    f"Epoch {epoch:02d} | "
                    f"Step {step:4d}/{args.steps_per_epoch} | "
                    f"Loss {loss.item():.4f} "
                    f"(bpr={loss_dict['bpr']:.4f}, reg={loss_dict['reg']:.4f})"
                )

        avg_loss = float(np.mean(epoch_losses))
        print(f"\n[Epoch {epoch}] Avg train loss: {avg_loss:.4f}")

        # ----------------- Evaluate on VAL -------------------
        metrics_val = eval_lightgcn_on_split(
            model,
            adj,
            loader,
            split="val",
            K=args.val_K,
            device=device,
            max_eval_users=args.val_max_users,
            batch_users=args.val_batch_users,
        )

        print(
            f"[Epoch {epoch}] VAL@{args.val_K} | "
            f"Precision: {metrics_val['precision']:.6f} | "
            f"Recall:    {metrics_val['recall']:.6f} | "
            f"HitRate:   {metrics_val['hit']:.6f} | "
            f"NDCG:      {metrics_val['ndcg']:.6f} | "
            f"MAP:       {metrics_val['map']:.6f}"
        )

        # ----------------- Early stopping --------------------
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

        # save "last" checkpoint every epoch (optional)
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

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------
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
        print("\n⚠ Training finished but no best_state was recorded (no epochs?).")


if __name__ == "__main__":
    main()

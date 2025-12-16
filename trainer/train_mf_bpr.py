# trainer/train_mf_bpr.py

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_utils.dataloader import TxtCFDataLoader
from src.models.MFBPR import MFBPR


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


# =========================
# Evaluation (LOO – giống các model khác)
# =========================
@torch.no_grad()
def eval_mf_on_split(
    model: MFBPR,
    loader: TxtCFDataLoader,
    split: str = "val",
    K: int = 20,
    batch_users: int = 1024,
    device: str = "cuda",
):
    assert split in {"val", "test"}
    model.eval()

    if split == "val":
        truth = loader.get_val_truth()
    else:
        truth = loader.get_test_truth()

    users = sorted(truth.keys())
    n_users = len(users)
    if n_users == 0:
        return dict(precision=0.0, recall=0.0, hit=0.0, ndcg=0.0, map=0.0)

    device_t = torch.device(
        "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    model = model.to(device_t)

    item_embs = model.item_emb.weight  # [I, d]

    total_precision = total_recall = total_hit = total_ndcg = total_map = 0.0
    import math

    for start in range(0, n_users, batch_users):
        end = min(start + batch_users, n_users)
        batch_users_list = users[start:end]

        u_tensor = torch.tensor(batch_users_list, dtype=torch.long, device=device_t)
        u_emb = model.user_emb(u_tensor)  # [B, d]

        scores = torch.matmul(u_emb, item_embs.t())  # [B, I]
        _, topk_idx = torch.topk(scores, K, dim=1)
        topk_idx = topk_idx.cpu().numpy()

        for i, u in enumerate(batch_users_list):
            gt = truth[u]
            rec_list = topk_idx[i].tolist()

            if gt in rec_list:
                total_hit += 1.0
                rank = rec_list.index(gt)
                total_recall += 1.0
                total_precision += 1.0 / K
                total_ndcg += 1.0 / math.log2(rank + 2)
                total_map += 1.0 / (rank + 1)

    return {
        "precision": total_precision / n_users,
        "recall": total_recall / n_users,
        "hit": total_hit / n_users,
        "ndcg": total_ndcg / n_users,
        "map": total_map / n_users,
    }


# =========================
# Sampling
# =========================
def detect_cols(df: pd.DataFrame):
    cols = set(df.columns)
    u_col = "u" if "u" in cols else ("user_idx" if "user_idx" in cols else None)
    v_col = "v" if "v" in cols else ("item_idx" if "item_idx" in cols else None)
    w_col = "weight" if "weight" in cols else ("time_weight" if "time_weight" in cols else None)
    assert u_col and v_col and w_col, f"Cannot detect columns in CSV: {df.columns.tolist()}"
    return u_col, v_col, w_col


def sample_batch(
    train_pos: Dict[int, List[int]],
    num_items: int,
    batch_size: int,
    use_time_decay: bool,
    pos_weight_map: Dict[tuple, float],
    device: torch.device,
):
    users, pos_items, neg_items, weights = [], [], [], []

    for _ in range(batch_size):
        u = random.choice(list(train_pos.keys()))
        pos_list = train_pos[u]
        pos = random.choice(pos_list)

        pos_set = set(pos_list)
        neg = random.randint(0, num_items - 1)
        while neg in pos_set:
            neg = random.randint(0, num_items - 1)

        users.append(u)
        pos_items.append(pos)
        neg_items.append(neg)

        if use_time_decay:
            weights.append(float(pos_weight_map.get((u, pos), 1.0)))
        else:
            weights.append(1.0)

    return (
        torch.tensor(users, dtype=torch.long, device=device),
        torch.tensor(pos_items, dtype=torch.long, device=device),
        torch.tensor(neg_items, dtype=torch.long, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser("Train MF-BPR")

    # Data
    p.add_argument("--data_dir", type=str, default="data/h_m")
    p.add_argument("--device", type=str, default="cuda")

    # Model
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps_per_epoch", type=int, default=800)
    p.add_argument("--clip_grad", type=float, default=1.0)

    # Time-decay
    p.add_argument("--use_time_decay", action="store_true")
    p.add_argument("--time_weight_csv", type=str, default=None)

    # Eval
    p.add_argument("--eval_K", type=int, default=20)
    p.add_argument("--eval_batch_users", type=int, default=1024)
    p.add_argument("--early_stop_patience", type=int, default=10)

    # Checkpoint
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--model_name", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_name = Path(args.data_dir).name
    tag = "td" if args.use_time_decay else "base"
    if args.model_name is None:
        args.model_name = f"mfbpr_{dataset_name}_{tag}"

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{args.model_name}_best.pt"
    last_path = ckpt_dir / f"{args.model_name}_last.pt"

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    loader = TxtCFDataLoader(args.data_dir, verbose=True)
    train_pos = loader.get_train_pos()
    num_users, num_items = loader.num_users, loader.num_items

    # Load time-decay weights
    pos_weight_map = {}
    if args.use_time_decay:
        assert args.time_weight_csv is not None, "time_weight_csv required"
        df = pd.read_csv(args.time_weight_csv)
        u_col, v_col, w_col = detect_cols(df)
        for u, v, w in zip(df[u_col], df[v_col], df[w_col]):
            pos_weight_map[(int(u), int(v))] = float(w)
        print(f"[TD] Loaded {len(pos_weight_map):,} time-decay edges")

    # Model
    model = MFBPR(num_users, num_items, args.emb_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_recall = -1.0
    best_epoch = -1
    best_state = None
    best_val_metrics = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step in range(1, args.steps_per_epoch + 1):
            users, pos, neg, w = sample_batch(
                train_pos, num_items, args.batch_size,
                args.use_time_decay, pos_weight_map, device
            )

            pos_scores, neg_scores = model(users, pos, neg)
            bpr = F.softplus(neg_scores - pos_scores)
            loss = (w * bpr).mean()

            reg = (
                model.user_emb(users).pow(2).sum(dim=1) +
                model.item_emb(pos).pow(2).sum(dim=1) +
                model.item_emb(neg).pow(2).sum(dim=1)
            ).mean() * 1e-6

            loss_total = loss + reg

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            if step == 1 or step % 50 == 0:
                print(
                    f"Epoch {epoch:02d} | Step {step:4d}/{args.steps_per_epoch} "
                    f"| Loss {loss_total.item():.4f}"
                )

        # Eval VAL
        metrics_val = eval_mf_on_split(
            model, loader, "val",
            args.eval_K, args.eval_batch_users, args.device
        )

        print(
            f"[VAL] epoch={epoch:02d} "
            f"precision={metrics_val['precision']:.4f} "
            f"recall={metrics_val['recall']:.4f} "
            f"hit={metrics_val['hit']:.4f} "
            f"ndcg={metrics_val['ndcg']:.4f} "
            f"map={metrics_val['map']:.4f}"
        )

        # Save LAST
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_users": num_users,
                "num_items": num_items,
                "emb_dim": args.emb_dim,
                "args": vars(args),
                "metrics_val": metrics_val,
                "best_recall": best_recall,
                "best_epoch": best_epoch,
            },
            last_path,
        )

        # Check best
        if metrics_val["recall"] > best_recall + 1e-6:
            best_recall = metrics_val["recall"]
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_metrics = metrics_val
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print("🛑 Early stopping.")
                break

    # Save BEST + eval TEST
    if best_state is not None:
        model.load_state_dict(best_state)
        metrics_test = eval_mf_on_split(
            model, loader, "test",
            args.eval_K, args.eval_batch_users, args.device
        )

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "num_users": num_users,
                "num_items": num_items,
                "emb_dim": args.emb_dim,
                "args": vars(args),
                "metrics_val": best_val_metrics,
                "metrics_test": metrics_test,
                "best_recall": best_recall,
                "best_epoch": best_epoch,
            },
            best_path,
        )

        print(
            f"[TEST] "
            f"precision={metrics_test['precision']:.4f} "
            f"recall={metrics_test['recall']:.4f} "
            f"hit={metrics_test['hit']:.4f} "
            f"ndcg={metrics_test['ndcg']:.4f} "
            f"map={metrics_test['map']:.4f}"
        )
        print(f"✅ Saved BEST to {best_path}")


if __name__ == "__main__":
    main()

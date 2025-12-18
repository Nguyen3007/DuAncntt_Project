#trainer/train_mf_bpr.py


import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data_utils.dataloader import TxtCFDataLoader
from src.models.MFBPR import MFBPR


# =========================
# Reproducibility
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
# Time-decay weight loader
# =========================
def _detect_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = set(df.columns)
    u_col = "u" if "u" in cols else ("user_idx" if "user_idx" in cols else None)
    v_col = "v" if "v" in cols else ("item_idx" if "item_idx" in cols else None)
    w_col = "weight" if "weight" in cols else ("time_weight" if "time_weight" in cols else None)
    if not (u_col and v_col and w_col):
        raise ValueError(f"Cannot detect columns in CSV: {df.columns.tolist()}")
    return u_col, v_col, w_col


def load_pos_weight_map(csv_path: str) -> Dict[Tuple[int, int], float]:
    df = pd.read_csv(csv_path)
    u_col, v_col, w_col = _detect_cols(df)
    # Using zip on series is fine here (one-time load).
    pos_weight_map: Dict[Tuple[int, int], float] = {}
    for u, v, w in zip(df[u_col], df[v_col], df[w_col]):
        pos_weight_map[(int(u), int(v))] = float(w)
    return pos_weight_map


# =========================
# Sampler
# =========================
class BPRSampler:
    """
    Fast BPR sampler.

    Why not precompute "all negatives per user" for H&M?
    - H&M has ~45k items and ~556k users; materializing (all_items - positives) for each user
      is O(U*I) in *memory*, totally infeasible.
    - Rejection sampling with a small positive set is fast in sparse settings.
    """

    def __init__(
        self,
        train_pos: Dict[int, List[int]],
        num_items: int,
        use_time_decay: bool = False,
        pos_weight_map: Optional[Dict[Tuple[int, int], float]] = None,
        cache_pos_sets: bool = False,
        pos_set_cache_limit: int = 200_000,
    ):
        self.train_pos = train_pos
        self.num_items = int(num_items)
        self.use_time_decay = bool(use_time_decay)
        self.pos_weight_map = pos_weight_map or {}

        # Cache user list ONCE (critical fix)
        self.users: np.ndarray = np.fromiter(train_pos.keys(), dtype=np.int64)

        # Optional cache for per-user positive sets (good for Vibrent; risky for full H&M)
        self.cache_pos_sets = bool(cache_pos_sets)
        self.pos_set_cache_limit = int(pos_set_cache_limit)
        self._pos_set_cache: Dict[int, set] = {}

    def _get_pos_set(self, u: int, pos_list: List[int]) -> set:
        if not self.cache_pos_sets:
            return set(pos_list)

        s = self._pos_set_cache.get(u)
        if s is not None:
            return s

        # Create and (maybe) cache
        s = set(pos_list)
        if len(self._pos_set_cache) < self.pos_set_cache_limit:
            self._pos_set_cache[u] = s
        return s

    def sample(self, batch_size: int, device: torch.device):
        # Vectorized user sampling like LightGCN/NGCF
        batch_users = np.random.choice(self.users, size=batch_size, replace=True)

        pos_items = np.empty(batch_size, dtype=np.int64)
        neg_items = np.empty(batch_size, dtype=np.int64)
        weights = np.ones(batch_size, dtype=np.float32)

        for i, u in enumerate(batch_users):
            u_int = int(u)
            pos_list = self.train_pos[u_int]
            p = random.choice(pos_list)

            pos_set = self._get_pos_set(u_int, pos_list)

            # rejection sampling for negative
            n = random.randint(0, self.num_items - 1)
            while n in pos_set:
                n = random.randint(0, self.num_items - 1)

            pos_items[i] = p
            neg_items[i] = n

            if self.use_time_decay:
                weights[i] = float(self.pos_weight_map.get((u_int, int(p)), 1.0))

        return (
            torch.as_tensor(batch_users, dtype=torch.long, device=device),
            torch.as_tensor(pos_items, dtype=torch.long, device=device),
            torch.as_tensor(neg_items, dtype=torch.long, device=device),
            torch.as_tensor(weights, dtype=torch.float32, device=device),
        )


# =========================
# Evaluation (LOO, same convention as repo)
# NOTE: User explicitly said: DO NOT mask train items (kept consistent with other models)
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

    truth = loader.get_val_truth() if split == "val" else loader.get_test_truth()
    users = sorted(truth.keys())
    n_users = len(users)
    if n_users == 0:
        return dict(precision=0.0, recall=0.0, hit=0.0, ndcg=0.0, map=0.0)

    device_t = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
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
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser("Train MF-BPR (optimized)")

    # Data
    p.add_argument("--data_dir", type=str, default="data/h_m")
    p.add_argument("--device", type=str, default="cuda")

    # Model
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--reg_weight", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps_per_epoch", type=int, default=400)
    p.add_argument("--clip_grad", type=float, default=5.0)

    # Time-decay
    p.add_argument("--use_time_decay", action="store_true")
    p.add_argument("--time_weight_csv", type=str, default=None)

    # Sampling optimization knobs
    p.add_argument("--cache_pos_sets", action="store_true",
                   help="Cache per-user positive sets (recommended for Vibrent; NOT recommended for full H&M).")
    p.add_argument("--pos_set_cache_limit", type=int, default=200_000)

    # Eval
    p.add_argument("--eval_K", type=int, default=20)
    p.add_argument("--eval_batch_users", type=int, default=4096)
    p.add_argument("--early_stop_patience", type=int, default=5)

    # Checkpoint
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--model_name", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =========================
# Main
# =========================
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

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\n=== Loading Data ===")
    loader = TxtCFDataLoader(args.data_dir, verbose=True)
    train_pos = loader.get_train_pos()
    num_users, num_items = loader.num_users, loader.num_items
    num_train_interactions = sum(len(v) for v in train_pos.values())
    print(f"Train interactions: {num_train_interactions:,}\n")

    # Load time-decay weights
    pos_weight_map: Dict[Tuple[int, int], float] = {}
    if args.use_time_decay:
        if args.time_weight_csv is None:
            args.time_weight_csv = os.path.join(args.data_dir, "train_time_weights.csv")

        if not os.path.exists(args.time_weight_csv):
            raise FileNotFoundError(f"time_weight_csv not found: {args.time_weight_csv}")

        print(f"[Time-Decay] Loading weights from: {args.time_weight_csv}")
        pos_weight_map = load_pos_weight_map(args.time_weight_csv)
        print(f"[Time-Decay] Loaded {len(pos_weight_map):,} weighted edges\n")

    # Sampler (critical for speed)
    sampler = BPRSampler(
        train_pos=train_pos,
        num_items=num_items,
        use_time_decay=args.use_time_decay,
        pos_weight_map=pos_weight_map,
        cache_pos_sets=args.cache_pos_sets,
        pos_set_cache_limit=args.pos_set_cache_limit,
    )

    # Model & optimizer
    print("=== Model Config ===")
    print(f"Embedding dim : {args.emb_dim}")
    print(f"Learning rate : {args.lr}")
    print(f"Weight decay  : {args.weight_decay}")
    print(f"Reg weight    : {args.reg_weight}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Steps/epoch   : {args.steps_per_epoch}")
    print(f"Grad clip     : {args.clip_grad}")
    print(f"Cache pos set : {args.cache_pos_sets} (limit={args.pos_set_cache_limit:,})")
    print("====================\n")

    model = MFBPR(num_users, num_items, args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_recall = -1.0
    best_epoch = -1
    best_state = None
    best_val_metrics = None
    bad_epochs = 0

    print("=== Training Start ===\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for step in range(1, args.steps_per_epoch + 1):
            users, pos, neg, w = sampler.sample(args.batch_size, device)

            pos_scores, neg_scores = model(users, pos, neg)

            bpr = F.softplus(neg_scores - pos_scores)  # stable -log(sigmoid(pos-neg))
            loss = (w * bpr).mean()

            # Explicit L2 on ego embeddings used in this batch (same idea as other trainers)
            if args.reg_weight > 0:
                reg = (
                    model.user_emb(users).pow(2).sum(dim=1)
                    + model.item_emb(pos).pow(2).sum(dim=1)
                    + model.item_emb(neg).pow(2).sum(dim=1)
                ).mean() * args.reg_weight
            else:
                reg = torch.zeros((), device=device)

            loss_total = loss + reg

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            if args.clip_grad is not None and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            epoch_losses.append(float(loss_total.item()))

            if step == 1 or step % 50 == 0:
                avg_loss = float(np.mean(epoch_losses))
                print(
                    f"Epoch {epoch:02d} | Step {step:4d}/{args.steps_per_epoch} "
                    f"| Loss {avg_loss:.4f} | BPR {float(loss.item()):.4f} | Reg {float(reg.item()):.6f}"
                )

        # Eval on VAL
        metrics_val = eval_mf_on_split(
            model=model,
            loader=loader,
            split="val",
            K=args.eval_K,
            batch_users=args.eval_batch_users,
            device=args.device,
        )

        print(
            f"\n[VAL] Epoch {epoch:02d} | "
            f"Precision@{args.eval_K}={metrics_val['precision']:.4f} | "
            f"Recall@{args.eval_K}={metrics_val['recall']:.4f} | "
            f"Hit@{args.eval_K}={metrics_val['hit']:.4f} | "
            f"NDCG@{args.eval_K}={metrics_val['ndcg']:.4f} | "
            f"MAP@{args.eval_K}={metrics_val['map']:.4f}\n"
        )

        # Save LAST checkpoint (consistent format)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_users": num_users,
                "num_items": num_items,
                "emb_dim": args.emb_dim,
                "best_recall": best_recall,
                "best_epoch": best_epoch,
                "args": vars(args),
                "metrics_val": metrics_val,
            },
            last_path,
        )

        # Track best by Recall@K (consistent with other trainers)
        if metrics_val["recall"] > best_recall + 1e-6:
            best_recall = float(metrics_val["recall"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_metrics = metrics_val
            bad_epochs = 0
            print(f"✅ New best @ epoch {epoch} | Recall@{args.eval_K}={best_recall:.4f}\n")
        else:
            bad_epochs += 1
            print(f"⚠️  No improvement {bad_epochs}/{args.early_stop_patience}\n")
            if bad_epochs >= args.early_stop_patience:
                print("🛑 Early stopping triggered.\n")
                break

    # Save BEST + evaluate TEST
    if best_state is not None:
        print("=== Final Evaluation on TEST set ===")
        model.load_state_dict(best_state, strict=True)

        metrics_test = eval_mf_on_split(
            model=model,
            loader=loader,
            split="test",
            K=args.eval_K,
            batch_users=args.eval_batch_users,
            device=args.device,
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
            f"\n[TEST] Best (Epoch {best_epoch}) | "
            f"Precision@{args.eval_K}={metrics_test['precision']:.4f} | "
            f"Recall@{args.eval_K}={metrics_test['recall']:.4f} | "
            f"Hit@{args.eval_K}={metrics_test['hit']:.4f} | "
            f"NDCG@{args.eval_K}={metrics_test['ndcg']:.4f} | "
            f"MAP@{args.eval_K}={metrics_test['map']:.4f}"
        )
        print(f"\n✅ Best model saved to: {best_path}")
        print(f"📁 Last checkpoint saved to: {last_path}\n")
    else:
        print("⚠️  No best model saved (training may have ended too early).")


if __name__ == "__main__":
    main()

# evaluate_mfbpr.py
import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import numpy as np
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_utils.dataloader import TxtCFDataLoader
from src.models.MFBPR import MFBPR


@torch.no_grad()
def eval_mfbpr_on_split(
    model: MFBPR,
    loader: TxtCFDataLoader,
    split: str,
    K: int,
    batch_users: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    LOO evaluation (consistent with your repo convention):
    - Each user has exactly 1 ground-truth item in val/test.
    - NO masking train items (per your standard across models).
    Metrics: Precision@K, Recall@K, Hit@K, NDCG@K, MAP@K
    """
    assert split in {"val", "test"}
    model.eval()

    truth = loader.get_val_truth() if split == "val" else loader.get_test_truth()
    users = sorted(truth.keys())
    n_users = len(users)
    if n_users == 0:
        return dict(precision=0.0, recall=0.0, hit=0.0, ndcg=0.0, map=0.0)

    item_embs = model.item_emb.weight  # [I, d]
    total_precision = total_recall = total_hit = total_ndcg = total_map = 0.0
    import math

    for start in range(0, n_users, batch_users):
        end = min(start + batch_users, n_users)
        batch_users_list = users[start:end]

        u = torch.tensor(batch_users_list, dtype=torch.long, device=device)
        u_emb = model.user_emb(u)  # [B, d]

        scores = torch.matmul(u_emb, item_embs.t())  # [B, I]
        _, topk = torch.topk(scores, K, dim=1)
        topk = topk.cpu().numpy()

        for i, uid in enumerate(batch_users_list):
            gt = truth[uid]
            rec = topk[i].tolist()

            if gt in rec:
                total_hit += 1.0
                rank = rec.index(gt)
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


def parse_args():
    p = argparse.ArgumentParser("Evaluate MF-BPR")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--batch_users", type=int, default=4096)

    # checkpoint
    p.add_argument("--checkpoint", type=str, required=True)

    # which split(s)
    p.add_argument("--split", type=str, default="both", choices=["val", "test", "both"])
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Load data
    loader = TxtCFDataLoader(args.data_dir, verbose=True)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint_path not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    num_users = int(ckpt["num_users"])
    num_items = int(ckpt["num_items"])
    emb_dim = int(ckpt["emb_dim"])

    model = MFBPR(num_users, num_items, emb_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    print(f"Loaded checkpoint: {ckpt_path.name}")
    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")

    # Eval
    if args.split in {"val", "both"}:
        m = eval_mfbpr_on_split(model, loader, "val", args.K, args.batch_users, device)
        print(
            f"[VAL]  Precision@{args.K}={m['precision']:.4f} | "
            f"Recall@{args.K}={m['recall']:.4f} | "
            f"Hit@{args.K}={m['hit']:.4f} | "
            f"NDCG@{args.K}={m['ndcg']:.4f} | "
            f"MAP@{args.K}={m['map']:.4f}"
        )

    if args.split in {"test", "both"}:
        m = eval_mfbpr_on_split(model, loader, "test", args.K, args.batch_users, device)
        print(
            f"[TEST] Precision@{args.K}={m['precision']:.4f} | "
            f"Recall@{args.K}={m['recall']:.4f} | "
            f"Hit@{args.K}={m['hit']:.4f} | "
            f"NDCG@{args.K}={m['ndcg']:.4f} | "
            f"MAP@{args.K}={m['map']:.4f}"
        )


if __name__ == "__main__":
    main()

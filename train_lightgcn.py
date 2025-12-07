# train_lightgcn.py

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch

from src.data_utils.dataloader import TxtCFDataLoader
from src.data_utils.graph_builder import GraphBuilder
from src.models.LightGCN import LightGCN


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
        # positive
        p = random.choice(pos_list)

        # negative
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGCN (H&M subset)")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/h_m")

    # Model
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--reg_weight", type=float, default=1e-3)

    # Train
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="0 = auto based on #interactions")
    parser.add_argument("--seed", type=int, default=42)

    # Device & save
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="checkpoints/lightgcn_hm.pt")

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
    train_pos = loader.get_train_pos()

    # Thống kê #interactions train
    num_train_interactions = sum(len(v) for v in train_pos.values())
    print(f"\n[Data] Train interactions: {num_train_interactions:,}")
    print(f"[Data] Users: {loader.num_users:,} | Items: {loader.num_items:,}")

    # 2) Build graph (trên đúng device)
    print("\n[Graph] Building normalized adjacency...")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) Training loop (mini-batch, dùng bpr_loss_slow)
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    else:
        steps_per_epoch = max(1, num_train_interactions // args.batch_size)

    print(
        f"[Train] steps_per_epoch = {steps_per_epoch} "
        f"(train_interactions={num_train_interactions:,}, batch_size={args.batch_size})\n"
    )

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

            # Gradient clipping để tránh explode
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()

            if step % 1 == 0:  # thay vì 50
                avg_step_loss = epoch_loss / step
                print(
                    f"Epoch {epoch:02d} | Step {step:5d}/{steps_per_epoch} "
                    f"| Loss {avg_step_loss:.4f} "
                    f"(bpr={log_dict['bpr']:.4f}, reg={log_dict['reg']:.4f})"
                )

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"\n[Epoch {epoch}] Avg Loss: {avg_epoch_loss:.4f}\n")

    # 5) Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": loader.num_users,
            "num_items": loader.num_items,
            "emb_dim": args.emb_dim,
            "n_layers": args.n_layers,
            "reg_weight": args.reg_weight,
        },
        args.save_path,
    )
    print("✅ Saved model to:", args.save_path)


if __name__ == "__main__":
    main()

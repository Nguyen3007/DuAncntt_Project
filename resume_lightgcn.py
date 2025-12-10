# resume_lightgcn.py
import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from src.data_utils.dataloader import TxtCFDataLoader
from src.data_utils.graph_builder import GraphBuilder
from src.data_utils.graph_builder_time_decay import TimeDecayGraphBuilder
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
# Evaluation (copy từ train_lightGCN_v2)
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
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume training LightGCN from an existing checkpoint "
                    "(support cả binary lẫn time-decay)."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path tới checkpoint LAST (.pt) đã train trước đó "
             "(ví dụ: checkpoints/lightgcn_hm_last_td.pt)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        required=True,
        help="Epoch tối đa muốn train đến (ví dụ: 80). "
             "Script sẽ tiếp tục từ epoch_saved + 1 tới max_epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda hoặc cpu (override nếu muốn).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Nếu None thì dùng giá trị lưu trong checkpoint args; "
             "nếu set thì override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Nếu None thì dùng seed trong checkpoint args.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args_cli = parse_args()

    ckpt_path = Path(args_cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1) Load checkpoint LAST
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt["args"]  # vars(args) khi train ban đầu

    # Lấy config gốc
    data_dir = ckpt_args.get("data_dir", "data/h_m")
    emb_dim = ckpt_args.get("emb_dim", 64)
    n_layers = ckpt_args.get("n_layers", 2)
    reg_weight = ckpt_args.get("reg_weight", 1e-3)
    lr = ckpt_args.get("lr", 5e-4)
    batch_size = ckpt_args.get("batch_size", 16384)
    steps_per_epoch = ckpt_args.get("steps_per_epoch", 800)
    val_K = ckpt_args.get("val_K", 20)
    val_batch_users = ckpt_args.get("val_batch_users", 1024)
    checkpoint_dir = ckpt_args.get("checkpoint_dir", "checkpoints")
    best_name = ckpt_args.get("best_name", "lightgcn_hm_best.pt")
    last_name = ckpt_args.get("last_name", "lightgcn_hm_last.pt")
    use_time_decay = ckpt_args.get("use_time_decay", False)
    time_weight_csv = ckpt_args.get("time_weight_csv", None)
    seed = args_cli.seed if args_cli.seed is not None else ckpt_args.get("seed", 42)
    early_stop_patience = (
        args_cli.early_stop_patience
        if args_cli.early_stop_patience is not None
        else ckpt_args.get("early_stop_patience", 3)
    )

    print("=== RESUME CONFIG (FROM CHECKPOINT) ===")
    print(f"Checkpoint path : {ckpt_path}")
    print(f"Data dir        : {data_dir}")
    print(f"emb_dim         : {emb_dim}")
    print(f"n_layers        : {n_layers}")
    print(f"reg_weight      : {reg_weight}")
    print(f"lr              : {lr}")
    print(f"batch_size      : {batch_size}")
    print(f"steps_per_epoch : {steps_per_epoch}")
    print(f"use_time_decay  : {use_time_decay}")
    print(f"time_weight_csv : {time_weight_csv}")
    print(f"best_name       : {best_name}")
    print(f"last_name       : {last_name}")
    print(f"max_epochs      : {args_cli.max_epochs}")
    print("=======================================\n")

    seed_everything(seed)

    device_str = args_cli.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, fallback to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print("Using device:", device)

    # 2) Load data
    loader = TxtCFDataLoader(data_dir, verbose=True)
    train_pos = loader.get_train_pos()

    num_train_interactions = sum(len(v) for v in train_pos.values())
    print(f"\n[Data] Train interactions: {num_train_interactions:,}")
    print(f"[Data] Users: {loader.num_users:,} | Items: {loader.num_items:,}\n")

    # 3) Build graph (binary hoặc time-decay)
    if use_time_decay:
        if time_weight_csv is None:
            time_weight_csv = os.path.join(data_dir, "train_time_weights.csv")
        print("[Graph] Building TIME-DECAY normalized adjacency...")
        gb = TimeDecayGraphBuilder(
            num_users=loader.num_users,
            num_items=loader.num_items,
            weight_csv=time_weight_csv,
            add_self_loop=False,
            verbose=True,
        )
    else:
        print("[Graph] Building BINARY normalized adjacency...")
        gb = GraphBuilder(
            num_users=loader.num_users,
            num_items=loader.num_items,
            train_user_items=train_pos,
        )

    adj = gb.build_normalized_adj(device=device)
    print("[Graph] Done.\n")

    # 4) Init model + optimizer, rồi load state
    model = LightGCN(
        num_users=loader.num_users,
        num_items=loader.num_items,
        emb_dim=emb_dim,
        n_layers=n_layers,
        reg_weight=reg_weight,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_recall = ckpt.get("best_recall", -1.0)
    best_epoch = ckpt.get("best_epoch", start_epoch - 1)

    print(
        f"🔁 Resume from epoch {start_epoch} "
        f"(current best Recall@{val_K} = {best_recall:.6f}, "
        f"best_epoch = {best_epoch})\n"
    )

    # 5) Load best_state nếu tồn tại
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir_path / best_name
    if best_path.exists():
        print(f"Found BEST checkpoint: {best_path}")
        ckpt_best = torch.load(best_path, map_location="cpu")
        best_state = ckpt_best["model_state_dict"]
        best_recall = ckpt_best.get("best_recall", best_recall)
        best_epoch = ckpt_best.get("epoch", best_epoch)
    else:
        print("No BEST checkpoint found, using LAST state as initial best.")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 6) Training loop tiếp tục
    if steps_per_epoch <= 0:
        steps_per_epoch = max(1, num_train_interactions // batch_size)

    patience_counter = 0

    for epoch in range(start_epoch, args_cli.max_epochs + 1):
        model.train()
        epoch_losses = []

        for step in range(1, steps_per_epoch + 1):
            users, pos_items, neg_items = sample_batch_bpr(
                train_user_items=train_pos,
                num_items=loader.num_items,
                batch_size=batch_size,
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

            if step == 1 or step % 50 == 0:
                avg_step_loss = float(np.mean(epoch_losses))
                print(
                    f"Epoch {epoch:02d} | Step {step:4d}/{steps_per_epoch} | "
                    f"Loss {avg_step_loss:.4f} "
                    f"(bpr={loss_dict['bpr']:.4f}, reg={loss_dict['reg']:.4f})"
                )

        avg_loss = float(np.mean(epoch_losses))
        print(f"\n[Epoch {epoch}] Avg train loss: {avg_loss:.4f}")

        # Eval trên VAL
        metrics_val = eval_lightgcn_on_split(
            model=model,
            adj=adj,
            loader=loader,
            device=device,
            split="val",
            K=val_K,
            batch_users=val_batch_users,
        )

        print(
            f"[Epoch {epoch}] VAL@{val_K} | "
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"⭐ New best model at epoch {epoch} "
                f"with Recall@{val_K} = {best_recall:.6f}"
            )
        else:
            patience_counter += 1
            print(
                f"[EarlyStop] No improvement in Recall@{val_K}. "
                f"patience = {patience_counter}/{early_stop_patience}"
            )

        # Save LAST checkpoint
        last_path = checkpoint_dir_path / last_name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_users": loader.num_users,
                "num_items": loader.num_items,
                "emb_dim": emb_dim,
                "n_layers": n_layers,
                "reg_weight": reg_weight,
                "best_recall": best_recall,
                "best_epoch": best_epoch,
                "args": ckpt_args,
            },
            last_path,
        )
        print(f" Saved last checkpoint to: {last_path}\n")

        if patience_counter >= early_stop_patience:
            print(
                f"⏹ Early stopping triggered at epoch {epoch} "
                f"(best epoch = {best_epoch}, best Recall@{val_K} = {best_recall:.6f})"
            )
            break

        # Save BEST checkpoint
        if best_state is not None:
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "num_users": loader.num_users,
                    "num_items": loader.num_items,
                    "emb_dim": emb_dim,
                    "n_layers": n_layers,
                    "reg_weight": reg_weight,
                    "best_recall": best_recall,
                    "args": ckpt_args,
                },
                best_path,
            )
            print(f"\n✅ Saved BEST model to: {best_path}")
        else:
            print("\n⚠ Training finished but no best_state was recorded.")


if __name__ == "__main__":
    main()

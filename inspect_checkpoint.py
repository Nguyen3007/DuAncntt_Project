# inspect_checkpoint.py

import argparse
import os
import sys
from pathlib import Path

import torch


def human_size(num_bytes: int) -> str:
    """Đổi byte -> chuỗi dễ đọc."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def detect_model_type(ckpt: dict) -> str:
    """
    Thử đoán checkpoint này thuộc LightGCN, NGCF hay unknown
    dựa trên keys trong state_dict.
    """
    state = (
        ckpt.get("model_state_dict")
        or ckpt.get("model_state")
        or ckpt.get("state_dict")
    )

    if state is None:
        return "unknown"

    keys = list(state.keys())

    # LightGCN chỉ có 1 embedding lớn, rất ít param khác
    if any("embedding.weight" in k for k in keys) and not any("W_gc_0" in k for k in keys):
        return "LightGCN"

    # NGCF có nhiều W_gc / W_bi / weight_layer_xx
    if any("W_gc" in k or "W_bi" in k for k in keys):
        return "NGCF"

    return "unknown"


def get_embedding_shape(ckpt: dict):
    """
    Tìm tensor embedding chính trong state_dict và trả về shape.
    (num_nodes, emb_dim) hoặc None nếu không tìm thấy.
    """
    state = (
        ckpt.get("model_state_dict")
        or ckpt.get("model_state")
        or ckpt.get("state_dict")
    )
    if state is None:
        return None

    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() == 2 and ("emb" in name or "embedding" in name):
            return tensor.shape  # (n_nodes, emb_dim)

    return None


def count_parameters(ckpt: dict) -> int:
    """
    Đếm tổng số tham số trong state_dict.
    """
    state = (
        ckpt.get("model_state_dict")
        or ckpt.get("model_state")
        or ckpt.get("state_dict")
    )
    if state is None:
        return 0

    total = 0
    for tensor in state.values():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel()
    return total


def inspect_checkpoint(path: Path, show_args: bool):
    if not path.exists():
        print(f"❌ Checkpoint not found: {path}")
        return

    print("=======================================")
    print(f"Checkpoint path : {path}")
    print(f"File size       : {human_size(path.stat().st_size)}")
    print("=======================================\n")

    ckpt = torch.load(path, map_location="cpu")

    # ---- Thông tin chung ----
    model_type = detect_model_type(ckpt)
    epoch = ckpt.get("epoch", None)
    best_epoch = ckpt.get("best_epoch", None)
    best_recall = ckpt.get("best_recall", None)

    num_users = ckpt.get("num_users", None)
    num_items = ckpt.get("num_items", None)

    print(f"Model type   : {model_type}")
    print(f"epoch        : {epoch}")
    print(f"best_epoch   : {best_epoch}")
    print(f"best_recall  : {best_recall}")
    print(f"num_users    : {num_users}")
    print(f"num_items    : {num_items}")

    # ---- Embedding shape ----
    emb_shape = get_embedding_shape(ckpt)
    if emb_shape is not None:
        n_nodes, emb_dim = emb_shape
        print(f"\nEmbedding shape : {emb_shape}  (nodes x dim)")
        if num_users is not None and num_items is not None:
            print(f"  → users + items (từ ckpt): {num_users} + {num_items} = {num_users + num_items}")
            print(f"  → nodes from emb shape   : {n_nodes}")
        print(f"Embedding dim   : {emb_dim}")
    else:
        print("\nEmbedding shape : (không tìm thấy trong state_dict)")

    # ---- Hyper-params từ args ----
    args = ckpt.get("args", None)
    if args is not None:
        # args lưu dạng dict (vars(args))
        print("\n=== Hyper-params chính (args) ===")
        # LightGCN/NGCF chung
        for key in [
            "data_dir",
            "emb_dim",
            "n_layers",
            "layer_sizes",
            "reg_weight",
            "mess_dropout",
            "lr",
            "batch_size",
            "steps_per_epoch",
            "val_K",
            "use_time_decay",
            "time_weight_csv",
            "device",
            "checkpoint_dir",
            "best_name",
            "last_name",
        ]:
            if key in args:
                print(f"{key:16s}: {args[key]}")
    else:
        print("\nKhông thấy field 'args' trong checkpoint.")

    # ---- Tổng số tham số ----
    total_params = count_parameters(ckpt)
    print(f"\nTotal parameters in state_dict: {total_params:,}")

    # ---- In full args nếu user yêu cầu ----
    if show_args and args is not None:
        print("\n=== FULL args dict ===")
        for k, v in args.items():
            print(f"{k:16s}: {v}")

    print("\n✅ Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect LightGCN/NGCF checkpoint (.pt) và in thông tin chi tiết."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Đường dẫn tới file .pt (best hoặc last).",
    )
    parser.add_argument(
        "--show_args",
        action="store_true",
        help="In toàn bộ args dict lưu trong checkpoint.",
    )

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)
    inspect_checkpoint(ckpt_path, show_args=args.show_args)


if __name__ == "__main__":
    main()

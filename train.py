# train.py
import argparse
from src.trainer import Trainer

def parse_args():
    p = argparse.ArgumentParser("Unified training entrypoint")
    p.add_argument(
        "--model",
        required=True,
        choices=["lightgcn", "ngcf", "als", "mfbpr"],
        help="Model to train"
    )
    args, unknown = p.parse_known_args()
    return args, unknown

def main():
    args, unknown = parse_args()
    trainer = Trainer(args.model)
    trainer.run(unknown)

if __name__ == "__main__":
    main()

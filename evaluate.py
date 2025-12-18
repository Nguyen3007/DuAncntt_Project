# evaluate.py
import argparse
from src.evaluator import Evaluator

def parse_args():
    p = argparse.ArgumentParser("Unified evaluation entrypoint")
    p.add_argument(
        "--model",
        required=True,
        choices=["lightgcn", "ngcf", "mfbpr"],
        help="Model to evaluate"
    )
    args, unknown = p.parse_known_args()
    return args, unknown

def main():
    args, unknown = parse_args()
    evaluator = Evaluator(args.model)
    evaluator.run(unknown)

if __name__ == "__main__":
    main()

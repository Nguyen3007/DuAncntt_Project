# src/evaluator.py
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVAL_RUNNERS = {
    "lightgcn": Path("evaluate/evaluate_lightgcn.py"),
    "ngcf": Path("evaluate/evaluate_ngcf.py"),
    "mfbpr": Path("evaluate/evaluate_mfbpr.py"),
}

class Evaluator:
    def __init__(self, model: str):
        self.model = model.lower()
        if self.model not in EVAL_RUNNERS:
            raise ValueError(f"Unknown model {model}")

    def run(self, passthrough_args):
        runner = EVAL_RUNNERS[self.model]
        cmd = [sys.executable, str(runner)] + passthrough_args

        print(">>> [Evaluator] Running:", " ".join(cmd))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)

        subprocess.run(
            cmd,
            check=True,
            cwd=str(ROOT),
            env=env,
        )

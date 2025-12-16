# src/trainer.py
import os
import subprocess
import sys
from pathlib import Path

# root project: DuAn_Project/
ROOT = Path(__file__).resolve().parents[1]

# mapping model -> script train hiện tại
RUNNERS = {
    "lightgcn": Path("trainer/train_lightGCN_v2.py"),
    "ngcf": Path("trainer/train_ngcf.py"),
    "als": Path("trainer/train_als.py"),
    "mfbpr": Path("trainer/train_mf_bpr.py"),
}

class Trainer:
    """
    Trainer chung – KHÔNG chứa thuật toán.
    Chỉ chịu trách nhiệm gọi đúng script train.
    """
    def __init__(self, model: str):
        self.model = model.lower()
        if self.model not in RUNNERS:
            raise ValueError(
                f"Unknown model '{model}'. Available: {list(RUNNERS.keys())}"
            )

    def run(self, passthrough_args):
        runner = RUNNERS[self.model]

        cmd = [sys.executable, str(runner)] + passthrough_args
        print(">>> [Trainer] Running:", " ".join(cmd))

        # đảm bảo import src.* hoạt động
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (
            os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else ""
        )

        subprocess.run(
            cmd,
            check=True,
            cwd=str(ROOT),
            env=env
        )

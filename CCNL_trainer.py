"""
Legacy trainer wrapper.
This delegates to the unified trainer in src/main_scripts/train.py, so
existing workflows calling CCNL_trainer.py keep working.
"""

from pathlib import Path
import argparse

from src.main_scripts.train import run_training


def parse_args():
    ap = argparse.ArgumentParser("CCNL trainer wrapper")
    ap.add_argument("--config", type=Path, default=Path("src/configs/training_config.yaml"))
    return ap.parse_args()


def main():
    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()

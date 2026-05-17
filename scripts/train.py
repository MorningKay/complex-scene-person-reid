"""Train a Re-ID model from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.engine import run_training
from reid.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Person Re-ID model.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    metrics = run_training(
        config=config,
        output_dir=output_dir,
        device=args.device,
        resume_checkpoint=args.resume,
    )
    print(f"metrics_json={output_dir / 'metrics.json'}", flush=True)
    print(f"avg_train_loss={metrics['avg_train_loss']:.6f}", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate a Re-ID checkpoint on Market-1501."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.engine import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Person Re-ID checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pth file.")
    parser.add_argument("--data-root", required=True, help="Market-1501 root directory.")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--distance",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Distance metric for retrieval ranking.",
    )
    parser.add_argument(
        "--max-query",
        type=int,
        default=None,
        help="Optional smoke-test limit for query images.",
    )
    parser.add_argument(
        "--max-gallery",
        type=int,
        default=None,
        help="Optional smoke-test limit for gallery images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_evaluation(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distance=args.distance,
        max_query=args.max_query,
        max_gallery=args.max_gallery,
    )
    print(f"eval_metrics_json={Path(args.output_dir) / 'eval_metrics.json'}", flush=True)
    print(f"rank1={metrics['rank1']:.6f}", flush=True)
    print(f"mAP={metrics['mAP']:.6f}", flush=True)


if __name__ == "__main__":
    main()

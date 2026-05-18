"""Evaluate a Re-ID checkpoint on a query/gallery dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.engine import run_evaluation
from reid.evaluation import (
    DEFAULT_RERANK_K1,
    DEFAULT_RERANK_K2,
    DEFAULT_RERANK_LAMBDA,
    DEFAULT_RERANK_NEIGHBOR_CHUNK_SIZE,
    DEFAULT_RERANK_QUERY_CHUNK_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Person Re-ID checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pth file.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory.")
    parser.add_argument(
        "--dataset-name",
        default="market1501",
        help="Dataset name, e.g. market1501 or msmt17_v1.",
    )
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
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=256,
        help="Number of query features evaluated per distance chunk.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply k-reciprocal re-ranking as a post-hoc evaluation step.",
    )
    parser.add_argument(
        "--rerank-k1",
        type=int,
        default=DEFAULT_RERANK_K1,
        help="k1 neighborhood size for k-reciprocal re-ranking.",
    )
    parser.add_argument(
        "--rerank-k2",
        type=int,
        default=DEFAULT_RERANK_K2,
        help="k2 query expansion neighborhood size for re-ranking.",
    )
    parser.add_argument(
        "--rerank-lambda",
        type=float,
        default=DEFAULT_RERANK_LAMBDA,
        help="Original-distance weight for re-ranking final distance.",
    )
    parser.add_argument(
        "--rerank-neighbor-chunk-size",
        type=int,
        default=DEFAULT_RERANK_NEIGHBOR_CHUNK_SIZE,
        help="Chunk size for all-sample top-k neighbor search during re-ranking.",
    )
    parser.add_argument(
        "--rerank-query-chunk-size",
        type=int,
        default=DEFAULT_RERANK_QUERY_CHUNK_SIZE,
        help="Query chunk size for final re-ranked query-gallery distance evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_evaluation(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distance=args.distance,
        max_query=args.max_query,
        max_gallery=args.max_gallery,
        query_chunk_size=args.query_chunk_size,
        rerank=args.rerank,
        rerank_k1=args.rerank_k1,
        rerank_k2=args.rerank_k2,
        rerank_lambda=args.rerank_lambda,
        rerank_neighbor_chunk_size=args.rerank_neighbor_chunk_size,
        rerank_query_chunk_size=args.rerank_query_chunk_size,
    )
    print(f"eval_metrics_json={Path(args.output_dir) / 'eval_metrics.json'}", flush=True)
    print(f"rank1={metrics['rank1']:.6f}", flush=True)
    print(f"mAP={metrics['mAP']:.6f}", flush=True)


if __name__ == "__main__":
    main()

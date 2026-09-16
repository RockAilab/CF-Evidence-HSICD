"""Command-line interface for inference-only sparse correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retained-evidence sparse-correction v4 inference")
    parser.add_argument("--base-evidence", type=Path, required=True, help="2D base evidence .npy")
    parser.add_argument("--boundary-evidence", type=Path, required=True, help="2D boundary evidence .npy")
    correction = parser.add_mutually_exclusive_group(required=True)
    correction.add_argument("--correction-probability", type=Path, help="2D learned correction probability .npy")
    correction.add_argument("--alpha-map", type=Path, help="2D ranked sparse alpha .npy")
    parser.add_argument("--correction-count", type=int, help="Unlabeled Top-K correction count")
    parser.add_argument("--prior", type=float, required=True, help="Unlabeled scene change prior")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def main() -> None:
    args = parse_args()
    base = load_npy(args.base_evidence)
    boundary = load_npy(args.boundary_evidence)
    probability = load_npy(args.correction_probability) if args.correction_probability else None
    alpha = load_npy(args.alpha_map) if args.alpha_map else None
    result = run_inference(
        base,
        boundary,
        args.prior,
        correction_probability=probability,
        correction_count=args.correction_count,
        alpha_map=alpha,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "prediction.npy", result.prediction)
    np.save(args.output_dir / "fused_score.npy", result.fused_score)
    np.save(args.output_dir / "alpha.npy", result.alpha)
    np.save(args.output_dir / "correction_mask.npy", result.correction_mask)
    metadata = {
        "shape": list(result.prediction.shape),
        "prior": result.prior,
        "route": result.route,
        "route_alpha": result.route_alpha,
        "correction_alpha": result.correction_alpha,
        "correction_pixels": int(result.correction_mask.sum()),
        "decision_threshold": result.threshold,
        "changed_pixels": int(result.prediction.sum()),
        "changed_ratio": float(result.prediction.mean()),
        "ground_truth_used": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

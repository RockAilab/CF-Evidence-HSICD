"""Command-line interface for counterfactual evidence inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selective evidence correction and rank-based inference")
    parser.add_argument("--base-evidence", type=Path, required=True, help="2D base counterfactual evidence .npy")
    parser.add_argument("--boundary-evidence", type=Path, required=True, help="2D boundary-spectral evidence .npy")
    correction = parser.add_mutually_exclusive_group(required=True)
    correction.add_argument("--correction-score", type=Path, help="2D correction score .npy")
    correction.add_argument("--alpha-map", type=Path, help="2D evidence-reliance coefficient .npy")
    parser.add_argument("--correction-count", type=int, help="Top-K correction count")
    parser.add_argument(
        "--estimated-prevalence",
        type=float,
        required=True,
        help="Label-free scene-level change prevalence",
    )
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
    correction_score = load_npy(args.correction_score) if args.correction_score else None
    alpha = load_npy(args.alpha_map) if args.alpha_map else None
    result = run_inference(
        base,
        boundary,
        args.estimated_prevalence,
        correction_score=correction_score,
        correction_count=args.correction_count,
        alpha_map=alpha,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "prediction.npy", result.prediction)
    np.save(args.output_dir / "corrected_evidence.npy", result.corrected_evidence)
    np.save(args.output_dir / "alpha.npy", result.alpha)
    np.save(args.output_dir / "correction_mask.npy", result.correction_mask)
    metadata = {
        "shape": list(result.prediction.shape),
        "estimated_prevalence": result.estimated_prevalence,
        "route": result.route,
        "alpha_route": result.alpha_route,
        "alpha_corr": result.alpha_corr,
        "correction_pixels": int(result.correction_mask.sum()),
        "decision_threshold": result.decision_threshold,
        "changed_pixels": int(result.prediction.sum()),
        "changed_ratio": float(result.prediction.mean()),
        "ground_truth_used": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

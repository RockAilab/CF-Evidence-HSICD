"""Run the bundled inference inputs for all three benchmark scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hsi_cd_inference import estimate_prevalence, percentile_rank, run_inference


MANIFEST = ROOT / "evidence" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all bundled HSI-CD inference scenes")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    summaries = []
    for scene in manifest["scenes"]:
        name = scene["name"]
        scene_dir = ROOT / "evidence" / name.lower()
        base_response = load_array(scene_dir / scene["base_response"])
        base = load_array(scene_dir / scene["base_evidence"])
        boundary = load_array(scene_dir / scene["boundary_evidence"])
        correction_score = load_array(scene_dir / scene["correction_score"])

        prevalence = estimate_prevalence(base_response)
        expected_prevalence = float(scene["estimated_prevalence"])
        if not np.isclose(prevalence.estimated_prevalence, expected_prevalence, rtol=0.0, atol=1e-15):
            raise RuntimeError(
                f"{name}: estimated prevalence {prevalence.estimated_prevalence} "
                f"does not match manifest value {expected_prevalence}"
            )
        if not np.array_equal(percentile_rank(base_response), base):
            raise RuntimeError(f"{name}: base evidence does not match the ranked base response")

        result = run_inference(
            base,
            boundary,
            prevalence.estimated_prevalence,
            correction_score=correction_score,
            correction_count=int(scene["correction_count"]),
        )

        output_dir = args.output_dir / name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "prediction.npy", result.prediction)
        np.save(output_dir / "corrected_evidence.npy", result.corrected_evidence)
        np.save(output_dir / "alpha.npy", result.alpha)
        np.save(output_dir / "correction_mask.npy", result.correction_mask)
        metadata = {
            "dataset": name,
            "data_file": scene["data_file"],
            "shape": list(result.prediction.shape),
            "estimated_prevalence": result.estimated_prevalence,
            "preliminary_prevalence": prevalence.preliminary_prevalence,
            "prevalence_regime": prevalence.regime,
            "partition_thresholds": prevalence.thresholds,
            "partition_ratios": prevalence.partition_ratios,
            "route": result.route,
            "correction_pixels": int(result.correction_mask.sum()),
            "decision_threshold": result.decision_threshold,
            "changed_pixels": int(result.prediction.sum()),
            "changed_ratio": float(result.prediction.mean()),
            "ground_truth_used": False,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summaries.append(metadata)
        print(
            f"{name}: prevalence={result.estimated_prevalence:.12f}, route={result.route}, "
            f"changed_pixels={metadata['changed_pixels']}, output={output_dir}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

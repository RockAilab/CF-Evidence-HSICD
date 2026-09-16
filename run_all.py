"""Run the bundled v4 inference inputs for all three benchmark scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hsi_cd_inference import run_inference


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
        base = load_array(scene_dir / scene["base_evidence"])
        boundary = load_array(scene_dir / scene["boundary_evidence"])
        probability = load_array(scene_dir / scene["correction_probability"])
        result = run_inference(
            base,
            boundary,
            float(scene["prior"]),
            correction_probability=probability,
            correction_count=int(scene["correction_count"]),
        )

        output_dir = args.output_dir / name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "prediction.npy", result.prediction)
        np.save(output_dir / "fused_score.npy", result.fused_score)
        np.save(output_dir / "alpha.npy", result.alpha)
        np.save(output_dir / "correction_mask.npy", result.correction_mask)
        metadata = {
            "dataset": name,
            "data_file": scene["data_file"],
            "shape": list(result.prediction.shape),
            "prior": result.prior,
            "route": result.route,
            "correction_pixels": int(result.correction_mask.sum()),
            "decision_threshold": result.threshold,
            "changed_pixels": int(result.prediction.sum()),
            "changed_ratio": float(result.prediction.mean()),
            "ground_truth_used": False,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summaries.append(metadata)
        print(
            f"{name}: shape={tuple(result.prediction.shape)}, "
            f"changed_pixels={metadata['changed_pixels']}, output={output_dir}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

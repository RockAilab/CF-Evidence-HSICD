import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from hsi_cd_inference import estimate_prevalence, percentile_rank, run_inference


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Farmland": {
        "route": "base_evidence",
        "changed_pixels": 18238,
        "correction_pixels": 37052,
        "decision_threshold": 0.7113635185546345,
        "prediction_hash": "7724845428cb9d41ed335c2f63828d3806492b5b1f6f3d6c16801bdd8702c896",
    },
    "Hermiston": {
        "route": "boundary_spectral",
        "changed_pixels": 16558,
        "correction_pixels": 10405,
        "decision_threshold": 0.7773137815278625,
        "prediction_hash": "77eb94153c8e30232feda77fab96ea4afc8992badfa13ca125b708b772074498",
    },
    "River": {
        "route": "boundary_spectral",
        "changed_pixels": 10273,
        "correction_pixels": 1615,
        "decision_threshold": 0.9079617744544105,
        "prediction_hash": "2db0c14d8a9eb7a1b95a4b2a19c26ea41d21a7d2f91a8ce4e87e40d561a2b36b",
    },
}


class BundledInferenceTests(unittest.TestCase):
    def test_bundled_scenes_match_reported_inference(self) -> None:
        manifest = json.loads((ROOT / "evidence" / "manifest.json").read_text(encoding="utf-8"))
        for scene in manifest["scenes"]:
            name = scene["name"]
            expected = EXPECTED[name]
            scene_dir = ROOT / "evidence" / name.lower()
            base_response = np.load(scene_dir / scene["base_response"], allow_pickle=False)
            base = np.load(scene_dir / scene["base_evidence"], allow_pickle=False)
            boundary = np.load(scene_dir / scene["boundary_evidence"], allow_pickle=False)
            correction_score = np.load(scene_dir / scene["correction_score"], allow_pickle=False)

            prevalence = estimate_prevalence(base_response)
            self.assertEqual(prevalence.estimated_prevalence, scene["estimated_prevalence"])
            np.testing.assert_array_equal(percentile_rank(base_response), base)
            result = run_inference(
                base,
                boundary,
                prevalence.estimated_prevalence,
                correction_score=correction_score,
                correction_count=scene["correction_count"],
            )

            with self.subTest(dataset=name):
                self.assertEqual(result.route, expected["route"])
                self.assertEqual(int(result.correction_mask.sum()), expected["correction_pixels"])
                self.assertEqual(int(result.prediction.sum()), expected["changed_pixels"])
                self.assertEqual(result.decision_threshold, expected["decision_threshold"])
                self.assertEqual(hashlib.sha256(result.prediction.tobytes()).hexdigest(), expected["prediction_hash"])


if __name__ == "__main__":
    unittest.main()

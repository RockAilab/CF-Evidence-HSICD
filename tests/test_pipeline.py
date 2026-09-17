import unittest

import numpy as np

from hsi_cd_inference.pipeline import percentile_rank, run_inference, topk_mask


class PipelineTests(unittest.TestCase):
    def test_percentile_rank_is_stable_for_ties(self) -> None:
        score = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        ranked = percentile_rank(score)
        expected = np.array([[2.0 / 3.0, 0.0], [1.0 / 3.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(ranked, expected)

    def test_topk_mask_has_exact_count(self) -> None:
        score = np.arange(12, dtype=np.float32).reshape(3, 4) / 11.0
        mask = topk_mask(score, 3)
        self.assertEqual(int(mask.sum()), 3)
        self.assertTrue(np.all(mask.reshape(-1)[-3:] == 1))

    def test_score_inference_uses_route_endpoints(self) -> None:
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        boundary = np.flip(base, axis=1).copy()
        score = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
        cases = (
            (0.30, "base_evidence", 1.0, 0.95),
            (0.10, "boundary_spectral", 0.0, 0.20),
        )
        for estimated_prevalence, route, alpha_route, alpha_corr in cases:
            with self.subTest(estimated_prevalence=estimated_prevalence):
                result = run_inference(
                    base,
                    boundary,
                    estimated_prevalence,
                    correction_score=score,
                    correction_count=4,
                )
                self.assertEqual(int(result.correction_mask.sum()), 4)
                self.assertEqual(result.route, route)
                self.assertEqual(result.alpha_route, alpha_route)
                self.assertEqual(result.alpha_corr, alpha_corr)
                np.testing.assert_allclose(
                    np.sort(np.unique(result.alpha)),
                    np.sort(np.asarray([alpha_route, alpha_corr], dtype=np.float32)),
                )
                self.assertEqual(result.prediction.dtype, np.uint8)

    def test_requires_exactly_one_correction_input(self) -> None:
        evidence = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            run_inference(evidence, evidence, 0.1)


if __name__ == "__main__":
    unittest.main()

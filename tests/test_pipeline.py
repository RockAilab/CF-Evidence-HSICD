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
        probability = np.arange(12, dtype=np.float32).reshape(3, 4) / 11.0
        mask = topk_mask(probability, 3)
        self.assertEqual(int(mask.sum()), 3)
        self.assertTrue(np.all(mask.reshape(-1)[-3:] == 1))

    def test_probability_inference_uses_protocol_endpoints(self) -> None:
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        boundary = np.flip(base, axis=1).copy()
        probability = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
        for prior, route_alpha, correction_alpha in ((0.30, 1.0, 0.95), (0.10, 0.0, 0.20)):
            with self.subTest(prior=prior):
                result = run_inference(
                    base,
                    boundary,
                    prior,
                    correction_probability=probability,
                    correction_count=4,
                )
                self.assertEqual(int(result.correction_mask.sum()), 4)
                self.assertEqual(result.route_alpha, route_alpha)
                self.assertEqual(result.correction_alpha, correction_alpha)
                np.testing.assert_allclose(
                    np.sort(np.unique(result.alpha)),
                    np.sort(np.asarray([route_alpha, correction_alpha], dtype=np.float32)),
                )
                self.assertEqual(result.prediction.dtype, np.uint8)

    def test_requires_exactly_one_correction_input(self) -> None:
        evidence = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            run_inference(evidence, evidence, 0.1)


if __name__ == "__main__":
    unittest.main()

"""Pure NumPy implementation of the original v4 sparse-correction inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ROUTE_PRIOR_THRESHOLD = 0.25
DENSE_CORRECTION_ALPHA = 0.95
BOUNDARY_CORRECTION_ALPHA = 0.20
PRIOR_MIN = 0.01
PRIOR_MAX = 0.65


@dataclass(frozen=True)
class InferenceResult:
    prediction: np.ndarray
    fused_score: np.ndarray
    alpha: np.ndarray
    correction_mask: np.ndarray
    threshold: float
    prior: float
    route: str
    route_alpha: float
    correction_alpha: float


def _map(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def percentile_rank(score: np.ndarray) -> np.ndarray:
    """Reproduce the stable rank transform used by the original protocol."""

    score = _map("score", score)
    flat = score.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    rank = np.empty_like(flat, dtype=np.float32)
    rank[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return rank.reshape(score.shape)


def topk_mask(probability: np.ndarray, count: int) -> np.ndarray:
    """Select exactly ``count`` pixels using the original argpartition rule."""

    probability = _map("correction_probability", probability)
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError("correction_probability must lie in [0, 1]")
    count = int(count)
    if count < 0 or count > probability.size:
        raise ValueError(f"correction_count must lie in [0, {probability.size}]")
    flat_mask = np.zeros(probability.size, dtype=np.float32)
    if count:
        indices = np.argpartition(probability.reshape(-1), -count)[-count:]
        flat_mask[indices] = 1.0
    return flat_mask.reshape(probability.shape)


def route_parameters(prior: float) -> tuple[str, float, float]:
    """Return route name, hard-route alpha, and correction alpha endpoint."""

    prior = float(prior)
    if not np.isfinite(prior) or prior < 0.0 or prior > 1.0:
        raise ValueError("prior must be a finite number in [0, 1]")
    if prior >= ROUTE_PRIOR_THRESHOLD:
        return "dense_base_cnn", 1.0, DENSE_CORRECTION_ALPHA
    return "sparse_boundary_spectral", 0.0, BOUNDARY_CORRECTION_ALPHA


def ranked_alpha_from_probability(
    probability: np.ndarray,
    correction_count: int,
    prior: float,
) -> tuple[np.ndarray, np.ndarray, str, float, float]:
    """Convert learned correction probabilities to the ranked sparse alpha."""

    route, route_alpha, correction_alpha = route_parameters(prior)
    mask = topk_mask(probability, correction_count)
    alpha = route_alpha + (correction_alpha - route_alpha) * mask
    return alpha.astype(np.float32), mask.astype(np.uint8), route, route_alpha, correction_alpha


def prior_quantile_decision(score: np.ndarray, prior: float) -> tuple[np.ndarray, float]:
    """Apply the original clipped-prior quantile and strict greater-than rule."""

    score = _map("fused_score", score)
    clipped_prior = float(np.clip(float(prior), PRIOR_MIN, PRIOR_MAX))
    threshold = float(np.quantile(score.reshape(-1), 1.0 - clipped_prior))
    prediction = (score > threshold).astype(np.uint8)
    return prediction, threshold


def run_inference(
    base_evidence: np.ndarray,
    boundary_evidence: np.ndarray,
    prior: float,
    *,
    correction_probability: np.ndarray | None = None,
    correction_count: int | None = None,
    alpha_map: np.ndarray | None = None,
) -> InferenceResult:
    """Fuse fixed v4 evidence and return a binary change map without GT."""

    base = _map("base_evidence", base_evidence)
    boundary = _map("boundary_evidence", boundary_evidence)
    if base.shape != boundary.shape:
        raise ValueError(f"evidence shapes differ: {base.shape} vs {boundary.shape}")
    if (correction_probability is None) == (alpha_map is None):
        raise ValueError("provide exactly one of correction_probability or alpha_map")

    route, route_alpha, correction_alpha = route_parameters(prior)
    if correction_probability is not None:
        if correction_count is None:
            raise ValueError("correction_count is required with correction_probability")
        probability = _map("correction_probability", correction_probability)
        if probability.shape != base.shape:
            raise ValueError(f"correction_probability shape differs: {probability.shape} vs {base.shape}")
        alpha, correction_mask, route, route_alpha, correction_alpha = ranked_alpha_from_probability(
            probability, correction_count, prior
        )
    else:
        if correction_count is not None:
            raise ValueError("correction_count cannot be used with alpha_map")
        alpha = _map("alpha_map", alpha_map)
        if alpha.shape != base.shape:
            raise ValueError(f"alpha_map shape differs: {alpha.shape} vs {base.shape}")
        if float(alpha.min()) < 0.0 or float(alpha.max()) > 1.0:
            raise ValueError("alpha_map must lie in [0, 1]")
        correction_mask = (np.abs(alpha - route_alpha) > 1e-6).astype(np.uint8)

    base_rank = percentile_rank(base)
    boundary_rank = percentile_rank(boundary)
    fused_score = (alpha * base_rank + (1.0 - alpha) * boundary_rank).astype(np.float32)
    prediction, threshold = prior_quantile_decision(fused_score, prior)
    return InferenceResult(
        prediction=prediction,
        fused_score=fused_score,
        alpha=alpha.astype(np.float32),
        correction_mask=correction_mask,
        threshold=threshold,
        prior=float(prior),
        route=route,
        route_alpha=route_alpha,
        correction_alpha=correction_alpha,
    )

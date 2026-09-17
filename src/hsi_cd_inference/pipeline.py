"""Selective evidence correction and rank-based change inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prevalence import select_evidence_route


BASE_CORRECTION_ALPHA = 0.95
BOUNDARY_CORRECTION_ALPHA = 0.20
DECISION_PREVALENCE_MIN = 0.01
DECISION_PREVALENCE_MAX = 0.65


@dataclass(frozen=True)
class InferenceResult:
    prediction: np.ndarray
    corrected_evidence: np.ndarray
    alpha: np.ndarray
    correction_mask: np.ndarray
    decision_threshold: float
    estimated_prevalence: float
    route: str
    alpha_route: float
    alpha_corr: float


def _map(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def percentile_rank(score: np.ndarray) -> np.ndarray:
    """Apply percentile-rank normalization used by the inference protocol."""

    score = _map("score", score)
    flat = score.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    rank = np.empty_like(flat, dtype=np.float32)
    rank[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return rank.reshape(score.shape)


def topk_mask(score: np.ndarray, count: int) -> np.ndarray:
    """Select exactly ``count`` correction scores with deterministic Top-K."""

    score = _map("correction_score", score)
    if float(score.min()) < 0.0 or float(score.max()) > 1.0:
        raise ValueError("correction_score must lie in [0, 1]")
    count = int(count)
    if count < 0 or count > score.size:
        raise ValueError(f"correction_count must lie in [0, {score.size}]")
    flat_mask = np.zeros(score.size, dtype=np.float32)
    if count:
        indices = np.argpartition(score.reshape(-1), -count)[-count:]
        flat_mask[indices] = 1.0
    return flat_mask.reshape(score.shape)


def evidence_reliance_parameters(estimated_prevalence: float) -> tuple[str, float, float]:
    """Return the evidence route and its correction-reliance endpoints."""

    route = select_evidence_route(estimated_prevalence)
    if route == "base_evidence":
        return route, 1.0, BASE_CORRECTION_ALPHA
    return route, 0.0, BOUNDARY_CORRECTION_ALPHA


def ranked_alpha_from_score(
    correction_score: np.ndarray,
    correction_count: int,
    estimated_prevalence: float,
) -> tuple[np.ndarray, np.ndarray, str, float, float]:
    """Convert correction-score ranking into selective evidence reliance."""

    route, alpha_route, alpha_corr = evidence_reliance_parameters(estimated_prevalence)
    mask = topk_mask(correction_score, correction_count)
    alpha = alpha_route + (alpha_corr - alpha_route) * mask
    return alpha.astype(np.float32), mask.astype(np.uint8), route, alpha_route, alpha_corr


def rank_based_change_decision(
    corrected_evidence: np.ndarray,
    estimated_prevalence: float,
) -> tuple[np.ndarray, float]:
    """Partition corrected evidence using the prevalence-conditioned quantile."""

    corrected_evidence = _map("corrected_evidence", corrected_evidence)
    pi_hat = float(np.clip(float(estimated_prevalence), DECISION_PREVALENCE_MIN, DECISION_PREVALENCE_MAX))
    decision_threshold = float(np.quantile(corrected_evidence.reshape(-1), 1.0 - pi_hat))
    prediction = (corrected_evidence > decision_threshold).astype(np.uint8)
    return prediction, decision_threshold


def run_inference(
    base_evidence: np.ndarray,
    boundary_evidence: np.ndarray,
    estimated_prevalence: float,
    *,
    correction_score: np.ndarray | None = None,
    correction_count: int | None = None,
    alpha_map: np.ndarray | None = None,
) -> InferenceResult:
    """Run ground-truth-independent counterfactual evidence inference."""

    base = _map("base_evidence", base_evidence)
    boundary = _map("boundary_evidence", boundary_evidence)
    if base.shape != boundary.shape:
        raise ValueError(f"evidence shapes differ: {base.shape} vs {boundary.shape}")
    if (correction_score is None) == (alpha_map is None):
        raise ValueError("provide exactly one of correction_score or alpha_map")

    route, alpha_route, alpha_corr = evidence_reliance_parameters(estimated_prevalence)
    if correction_score is not None:
        if correction_count is None:
            raise ValueError("correction_count is required with correction_score")
        score = _map("correction_score", correction_score)
        if score.shape != base.shape:
            raise ValueError(f"correction_score shape differs: {score.shape} vs {base.shape}")
        alpha, correction_mask, route, alpha_route, alpha_corr = ranked_alpha_from_score(
            score, correction_count, estimated_prevalence
        )
    else:
        if correction_count is not None:
            raise ValueError("correction_count cannot be used with alpha_map")
        alpha = _map("alpha_map", alpha_map)
        if alpha.shape != base.shape:
            raise ValueError(f"alpha_map shape differs: {alpha.shape} vs {base.shape}")
        if float(alpha.min()) < 0.0 or float(alpha.max()) > 1.0:
            raise ValueError("alpha_map must lie in [0, 1]")
        correction_mask = (np.abs(alpha - alpha_route) > 1e-6).astype(np.uint8)

    base_rank = percentile_rank(base)
    boundary_rank = percentile_rank(boundary)
    corrected_evidence = (alpha * base_rank + (1.0 - alpha) * boundary_rank).astype(np.float32)
    prediction, decision_threshold = rank_based_change_decision(corrected_evidence, estimated_prevalence)
    return InferenceResult(
        prediction=prediction,
        corrected_evidence=corrected_evidence,
        alpha=alpha.astype(np.float32),
        correction_mask=correction_mask,
        decision_threshold=decision_threshold,
        estimated_prevalence=float(estimated_prevalence),
        route=route,
        alpha_route=alpha_route,
        alpha_corr=alpha_corr,
    )

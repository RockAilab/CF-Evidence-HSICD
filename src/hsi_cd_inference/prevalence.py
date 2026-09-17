"""Deterministic label-free change-prevalence estimation and routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_li, threshold_otsu, threshold_yen


SPARSE_TEST_FACTOR = 0.75
YEN_MULTIPLIER = 1.10
YEN_CAP = 0.11
MEDIUM_OTSU_BOUNDARY = 0.22
DENSE_OTSU_MULTIPLIER = 0.95
PREVALENCE_MIN = 0.02
PREVALENCE_MAX = 0.55
ROUTE_PREVALENCE_THRESHOLD = 0.25


@dataclass(frozen=True)
class PrevalenceEstimate:
    estimated_prevalence: float
    preliminary_prevalence: float
    regime: str
    thresholds: dict[str, float]
    partition_ratios: dict[str, float]


def _base_response(value: np.ndarray) -> np.ndarray:
    response = np.asarray(value, dtype=np.float32)
    if response.ndim != 2:
        raise ValueError(f"base_response must be a 2D array; got shape {response.shape}")
    if not np.isfinite(response).all():
        raise ValueError("base_response contains non-finite values")
    if float(response.max()) <= float(response.min()):
        raise ValueError("base_response must contain more than one distinct value")
    return response


def estimate_partition_ratios(base_response: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Compute Otsu, Yen, and Li partitions and their foreground ratios."""

    response = _base_response(base_response)
    finite = response.reshape(-1)
    thresholds = {
        "otsu": float(threshold_otsu(finite)),
        "yen": float(threshold_yen(finite)),
        "li": float(threshold_li(finite)),
    }
    ratios = {name: float((response > threshold).mean()) for name, threshold in thresholds.items()}
    return thresholds, ratios


def estimate_prevalence(base_response: np.ndarray) -> PrevalenceEstimate:
    """Apply the fixed regime operator to obtain label-free prevalence."""

    thresholds, ratios = estimate_partition_ratios(base_response)
    if ratios["yen"] < SPARSE_TEST_FACTOR * ratios["otsu"]:
        preliminary = min(YEN_CAP, YEN_MULTIPLIER * ratios["yen"])
        regime = "sparse_yen"
    elif ratios["otsu"] < MEDIUM_OTSU_BOUNDARY:
        preliminary = ratios["li"]
        regime = "medium_li"
    else:
        preliminary = DENSE_OTSU_MULTIPLIER * ratios["otsu"]
        regime = "dense_otsu"
    estimated = float(np.clip(preliminary, PREVALENCE_MIN, PREVALENCE_MAX))
    return PrevalenceEstimate(
        estimated_prevalence=estimated,
        preliminary_prevalence=float(preliminary),
        regime=regime,
        thresholds=thresholds,
        partition_ratios=ratios,
    )


def select_evidence_route(estimated_prevalence: float) -> str:
    """Select the scene-level evidence route from estimated prevalence."""

    value = float(estimated_prevalence)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("estimated_prevalence must be a finite number in [0, 1]")
    if value >= ROUTE_PREVALENCE_THRESHOLD:
        return "base_evidence"
    return "boundary_spectral"

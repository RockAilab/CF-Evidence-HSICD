#!/usr/bin/env python3
"""Regenerate fixed-inference supplementary audits and sensitivity plots.

This script does not train a model, select parameters, or modify bundled
evidence and reference outputs. Ground truth is loaded only after all reported
fixed-inference outputs have been reconstructed and verified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from skimage.segmentation import find_boundaries


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsi_cd_inference import estimate_prevalence, percentile_rank, run_inference  # noqa: E402
from hsi_cd_inference.pipeline import rank_based_change_decision, topk_mask  # noqa: E402


ROUTING_THRESHOLDS = (0.20, 0.225, 0.25, 0.275, 0.30)
AMPLITUDE_SCALES = (0.75, 1.00, 1.25)
OFFICIAL_ROUTE_THRESHOLD = 0.25
RHO_BASE = 0.05
RHO_BOUNDARY = 0.20
EXPECTED_ROUTES = {
    "Farmland": "base_evidence",
    "Hermiston": "boundary_spectral",
    "River": "boundary_spectral",
}
EXPECTED_SCORE_HASHES = {
    "Farmland": "771893fd2da67c60ce65cb8f6f7eaefba91fcfa6fd159bba5ff423d8eb4c04a3",
    "Hermiston": "ac72ef1311a0fb85ad42ffbd0744a226f854a7f74f3615cc06b8011e7adb4276",
    "River": "f6e09e58edac5a252f3704bd077fb5bbe1ce33b5178ce0306f70ce652d23453c",
}
AUDIT_FIELDS = [
    "Dataset", "Data source", "Code entry point", "GT usage", "Parameter source",
    "Output file", "Check", "Expected value", "Observed value", "Mismatch", "Status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate fixed-inference supplementary audits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "supplementary_outputs")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    total = tp + tn + fp + fn
    oa = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
    expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total)
    kappa = (oa - expected) / max(1.0 - expected, np.finfo(float).eps)
    return {"OA": oa, "F1": f1, "Kappa": kappa}


def inference_with_perturbation(
    base: np.ndarray,
    boundary: np.ndarray,
    score: np.ndarray,
    correction_count: int,
    prevalence: float,
    route_threshold: float = OFFICIAL_ROUTE_THRESHOLD,
    amplitude_scale: float = 1.0,
) -> dict[str, object]:
    route_base = prevalence >= route_threshold
    route = "base_evidence" if route_base else "boundary_spectral"
    alpha_route = 1.0 if route_base else 0.0
    rho = RHO_BASE if route_base else RHO_BOUNDARY
    alpha_corr = 1.0 - amplitude_scale * rho if route_base else amplitude_scale * rho
    if not 0.0 <= alpha_corr <= 1.0:
        raise ValueError(f"Invalid correction endpoint: {alpha_corr}")
    mask = topk_mask(score, correction_count).astype(np.uint8)
    alpha = alpha_route + (alpha_corr - alpha_route) * mask
    base_rank = percentile_rank(base)
    boundary_rank = percentile_rank(boundary)
    corrected = (alpha * base_rank + (1.0 - alpha) * boundary_rank).astype(np.float32)
    prediction, threshold = rank_based_change_decision(corrected, prevalence)
    routed = base_rank if route_base else boundary_rank
    route_prediction, route_decision_threshold = rank_based_change_decision(routed, prevalence)
    return {
        "route": route,
        "mask": mask,
        "corrected": corrected,
        "prediction": prediction,
        "threshold": threshold,
        "route_prediction": route_prediction,
        "route_decision_threshold": route_decision_threshold,
    }


def load_fixed_inputs() -> list[dict[str, object]]:
    manifest = json.loads((ROOT / "evidence/manifest.json").read_text(encoding="utf-8"))
    scenes: list[dict[str, object]] = []
    for entry in manifest["scenes"]:
        name = entry["name"]
        stem = name.lower()
        evidence_dir = ROOT / "evidence" / stem
        reference_dir = ROOT / "audit_reference" / stem
        scene = {
            "name": name,
            "stem": stem,
            "entry": entry,
            "base_response": np.load(evidence_dir / entry["base_response"], allow_pickle=False),
            "base": np.load(evidence_dir / entry["base_evidence"], allow_pickle=False),
            "boundary": np.load(evidence_dir / entry["boundary_evidence"], allow_pickle=False),
            "score": np.load(evidence_dir / entry["correction_score"], allow_pickle=False),
            "reference_route": np.load(reference_dir / "route_prediction.npy", allow_pickle=False),
            "reference_mask": np.load(reference_dir / "correction_mask.npy", allow_pickle=False),
            "reference_corrected": np.load(reference_dir / "corrected_evidence.npy", allow_pickle=False),
            "reference_prediction": np.load(reference_dir / "prediction.npy", allow_pickle=False),
        }
        shapes = {value.shape for value in scene.values() if isinstance(value, np.ndarray)}
        if len(shapes) != 1:
            raise RuntimeError(f"{name}: incompatible array shapes: {sorted(shapes)}")
        scenes.append(scene)
    return scenes


def audit_row(
    scene: dict[str, object], check: str, output_file: str, expected: object,
    observed: object, mismatch: object, status: str = "Reproduced",
) -> dict[str, object]:
    entry = scene["entry"]
    return {
        "Dataset": scene["name"],
        "Data source": f"evidence/{scene['stem']}/ and {entry['data_file']}",
        "Code entry point": "scripts/generate_supplementary.py",
        "GT usage": "No",
        "Parameter source": "evidence/manifest.json and fixed inference implementation",
        "Output file": output_file,
        "Check": check,
        "Expected value": expected,
        "Observed value": observed,
        "Mismatch": mismatch,
        "Status": status,
    }


def verify_fixed_inference(
    scenes: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prevalence_rows: list[dict[str, object]] = []
    consistency_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for scene in scenes:
        name = scene["name"]
        stem = scene["stem"]
        entry = scene["entry"]
        estimate = estimate_prevalence(scene["base_response"])
        expected_prevalence = float(entry["estimated_prevalence"])
        prevalence_delta = float(estimate.estimated_prevalence - expected_prevalence)
        base_rank_mismatch = int(np.count_nonzero(percentile_rank(scene["base_response"]) != scene["base"]))
        if prevalence_delta != 0.0 or base_rank_mismatch:
            raise RuntimeError(f"{name}: prevalence or base-rank reproduction failed")

        result = run_inference(
            scene["base"], scene["boundary"], estimate.estimated_prevalence,
            correction_score=scene["score"], correction_count=int(entry["correction_count"]),
        )
        routed = percentile_rank(scene["base"] if result.route == "base_evidence" else scene["boundary"])
        route_prediction, _ = rank_based_change_decision(routed, estimate.estimated_prevalence)
        route_mismatch = int(np.count_nonzero(route_prediction != scene["reference_route"]))
        mask_mismatch = int(np.count_nonzero(result.correction_mask != scene["reference_mask"]))
        prediction_mismatch = int(np.count_nonzero(result.prediction != scene["reference_prediction"]))
        evidence_error = np.abs(
            result.corrected_evidence.astype(np.float64) - scene["reference_corrected"].astype(np.float64)
        )
        evidence_mismatch = int(np.count_nonzero(evidence_error > 1e-7))
        score_hash = array_hash(scene["score"])
        score_mismatch = int(score_hash != EXPECTED_SCORE_HASHES[name])
        route_value_mismatch = int(result.route != EXPECTED_ROUTES[name])
        if any((route_mismatch, mask_mismatch, prediction_mismatch, evidence_mismatch,
                score_mismatch, route_value_mismatch)):
            raise RuntimeError(f"{name}: fixed-inference audit failed")

        ratios = estimate.partition_ratios
        prevalence_rows.append({
            "Dataset": name,
            "r_Otsu": ratios["otsu"],
            "r_Yen": ratios["yen"],
            "r_Li": ratios["li"],
            "Regime": estimate.regime,
            "Estimated prevalence": estimate.estimated_prevalence,
            "Selected evidence": result.route,
        })
        consistency_rows.append({
            "Dataset": name,
            "Prevalence delta": prevalence_delta,
            "Route prediction mismatch": route_mismatch,
            "Correction score mismatch": score_mismatch,
            "Correction mask mismatch": mask_mismatch,
            "Corrected evidence mismatch (>1e-7)": evidence_mismatch,
            "Corrected prediction mismatch": prediction_mismatch,
        })
        audit_rows.extend([
            audit_row(scene, "Prevalence", "evidence/manifest.json", expected_prevalence,
                      estimate.estimated_prevalence, prevalence_delta),
            audit_row(scene, "Scene-level routing", f"audit_reference/{stem}/route_prediction.npy",
                      EXPECTED_ROUTES[name], result.route, route_mismatch),
            audit_row(scene, "Correction score", f"evidence/{stem}/{entry['correction_score']}",
                      EXPECTED_SCORE_HASHES[name], score_hash, score_mismatch),
            audit_row(scene, "Correction mask", f"audit_reference/{stem}/correction_mask.npy",
                      "Exact reference map", "Exact match", f"{mask_mismatch} pixels"),
            audit_row(scene, "Corrected evidence", f"audit_reference/{stem}/corrected_evidence.npy",
                      "Tolerance 1e-7", f"max abs error {float(evidence_error.max(initial=0.0)):.3g}",
                      f"{evidence_mismatch} pixels"),
            audit_row(scene, "Final prediction", f"audit_reference/{stem}/prediction.npy",
                      "Exact reference map", "Exact match", f"{prediction_mismatch} pixels"),
        ])
        scene["result"] = result

    farmland = next(scene for scene in scenes if scene["name"] == "Farmland")
    audit_rows.append(audit_row(
        farmland, "Historical initialization", "evidence/farmland/base_response.npy",
        "Fixed-input reproduction only", "Fixed-input reproduction only", "N/A",
    ))
    return prevalence_rows, consistency_rows, audit_rows


def load_ground_truth(scenes: list[dict[str, object]]) -> None:
    for scene in scenes:
        path = ROOT / scene["entry"]["data_file"]
        ground_truth = np.asarray(loadmat(path)["GT"], dtype=np.uint8)
        if ground_truth.shape != scene["result"].prediction.shape:
            raise RuntimeError(f"{scene['name']}: ground-truth shape mismatch")
        scene["ground_truth"] = ground_truth


def posthoc_tables(
    scenes: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    correction_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    amplitude_rows: list[dict[str, object]] = []
    for scene in scenes:
        result = scene["result"]
        gt = scene["ground_truth"].astype(bool)
        route_pred = result.route_prediction.astype(bool) if hasattr(result, "route_prediction") else None
        if route_pred is None:
            routed = percentile_rank(scene["base"] if result.route == "base_evidence" else scene["boundary"])
            route_pred, _ = rank_based_change_decision(routed, result.estimated_prevalence)
            route_pred = route_pred.astype(bool)
        corrected = result.prediction.astype(bool)
        support = result.correction_mask.astype(bool)
        changed = route_pred != corrected
        repair = changed & (corrected == gt)
        damage = changed & (route_pred == gt)
        correction_rows.append({
            "Dataset": scene["name"],
            "Support (%)": 100.0 * support.mean(),
            "Decision change (%)": 100.0 * changed.mean(),
            "Repair": int(repair.sum()),
            "Damage": int(damage.sum()),
        })

        prevalence = float(scene["entry"]["estimated_prevalence"])
        count = int(scene["entry"]["correction_count"])
        for threshold in ROUTING_THRESHOLDS:
            perturbed = inference_with_perturbation(
                scene["base"], scene["boundary"], scene["score"], count, prevalence, threshold,
            )
            values = metrics(perturbed["prediction"], scene["ground_truth"])
            routing_rows.append({
                "Dataset": scene["name"], "Routing threshold": threshold,
                "Selected evidence": perturbed["route"], **values,
            })
        for scale in AMPLITUDE_SCALES:
            perturbed = inference_with_perturbation(
                scene["base"], scene["boundary"], scene["score"], count, prevalence,
                OFFICIAL_ROUTE_THRESHOLD, scale,
            )
            values = metrics(perturbed["prediction"], scene["ground_truth"])
            amplitude_rows.append({
                "Dataset": scene["name"], "Amplitude scale": scale,
                "F1": values["F1"], "Kappa": values["Kappa"],
                "Decision change (%)": 100.0 * np.mean(perturbed["prediction"] != route_pred),
            })
    return correction_rows, routing_rows, amplitude_rows


def plot_sensitivity(
    output_dir: Path, routing: list[dict[str, object]], amplitude: list[dict[str, object]],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"Farmland": "#0072B2", "Hermiston": "#D55E00", "River": "#009E73"}
    markers = {"Farmland": "o", "Hermiston": "s", "River": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    for dataset in colors:
        rows = [row for row in routing if row["Dataset"] == dataset]
        x = [row["Routing threshold"] for row in rows]
        for axis, key in zip(axes, ("OA", "F1", "Kappa")):
            axis.plot(x, [100.0 * row[key] for row in rows], marker=markers[dataset],
                      color=colors[dataset], label=dataset if key == "OA" else None)
    for axis, label in zip(axes, ("OA (%)", "F1 (%)", "Kappa (%)")):
        axis.axvline(OFFICIAL_ROUTE_THRESHOLD, color="#555555", linestyle="--", linewidth=1)
        axis.set_xlabel(r"Routing threshold $\tau_r$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "routing_threshold_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(figures / "routing_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    for dataset in colors:
        rows = [row for row in amplitude if row["Dataset"] == dataset]
        x = [row["Amplitude scale"] for row in rows]
        keys = ("F1", "Kappa", "Decision change (%)")
        for axis, key in zip(axes, keys):
            values = [100.0 * row[key] for row in rows] if key != "Decision change (%)" else [row[key] for row in rows]
            axis.plot(x, values, marker=markers[dataset], color=colors[dataset],
                      label=dataset if key == "F1" else None)
    for axis, label in zip(axes, ("F1 (%)", "Kappa (%)", "Decision change (%)")):
        axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xlabel("Joint amplitude scale")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "correction_amplitude_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(figures / "correction_amplitude_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def densest_support_window(support: np.ndarray, size: int = 96) -> tuple[int, int, int, int]:
    height, width = support.shape
    window_h, window_w = min(size, height), min(size, width)
    integral = np.pad(support.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (integral[window_h:, window_w:] - integral[:-window_h, window_w:]
            - integral[window_h:, :-window_w] + integral[:-window_h, :-window_w])
    y0, x0 = (int(value) for value in np.unravel_index(int(np.argmax(sums)), sums.shape))
    return y0, y0 + window_h, x0, x0 + window_w


def plot_local_cases(output_dir: Path, scenes: list[dict[str, object]]) -> None:
    selected = [next(scene for scene in scenes if scene["name"] == name) for name in ("Hermiston", "River")]
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.7), constrained_layout=True)
    for column, title in enumerate(("Routed prediction", "Correction score / support", "Corrected prediction", "GT")):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")
    for row, scene in enumerate(selected):
        result = scene["result"]
        support = result.correction_mask.astype(np.uint8)
        y0, y1, x0, x1 = densest_support_window(support)
        routed = percentile_rank(scene["base"] if result.route == "base_evidence" else scene["boundary"])
        route_prediction, _ = rank_based_change_decision(routed, result.estimated_prevalence)
        gt = scene["ground_truth"].astype(np.uint8)
        boundary = find_boundaries(gt.astype(bool), connectivity=2, mode="inner")
        region = np.s_[y0:y1, x0:x1]
        axes[row, 0].imshow(route_prediction[region], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 1].imshow(scene["score"][region], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        local_support = support[region]
        if local_support.min() != local_support.max():
            axes[row, 1].contour(local_support, levels=[0.5], colors=["#ff3d00"], linewidths=0.65)
        axes[row, 2].imshow(result.prediction[region], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 3].imshow(gt[region], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        local_boundary = boundary[region]
        if local_boundary.any():
            axes[row, 3].contour(local_boundary, levels=[0.5], colors=["#00b8d4"], linewidths=0.55)
        axes[row, 0].set_ylabel(f"{scene['name']}\n({y0}:{y1}, {x0}:{x1})", fontsize=9)
        for column in range(4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    output = output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "local_correction_cases.pdf", bbox_inches="tight")
    fig.savefig(output / "local_correction_cases.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scenes = load_fixed_inputs()
    prevalence, consistency, audit = verify_fixed_inference(scenes)
    load_ground_truth(scenes)
    correction, routing, amplitude = posthoc_tables(scenes)
    tables = args.output_dir / "tables"
    write_csv(tables / "prevalence_routing_audit.csv", prevalence)
    write_csv(tables / "prediction_consistency.csv", consistency)
    write_csv(tables / "correction_decision_audit.csv", correction)
    write_csv(tables / "routing_threshold_sensitivity.csv", routing)
    write_csv(tables / "correction_amplitude_sensitivity.csv", amplitude)
    write_csv(tables / "reproducibility_audit.csv", audit, AUDIT_FIELDS)
    plot_sensitivity(args.output_dir, routing, amplitude)
    plot_local_cases(args.output_dir, scenes)
    summary = {
        "scope": "Fixed inference reproduction; no training or parameter selection",
        "ground_truth_role": "Post-hoc metrics and visualization only",
        "farmland_historical_initialization": "Fixed-input reproduction only",
        "prediction_mismatch": {
            row["Dataset"]: row["Corrected prediction mismatch"] for row in consistency
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

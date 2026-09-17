# Counterfactual Evidence Inference for Label-Free Hyperspectral Change Detection

This repository provides the inference implementation and fixed evidence required to reproduce the reported binary change maps on Farmland, Hermiston, and River.

## Paper Reproduction Version

The results reported in the manuscript and supplementary material correspond
to:

- Release tag: `paper-submission-v2`
- Commit: `14ad04c0a5995c1b150f19bde0e1a9442c22b92e`

For exact reproduction of the submitted results, use this tagged version.

## Reproduction Scope

The repository supports fixed inference reproduction. It includes the benchmark data, pre-rank base responses, base counterfactual evidence, boundary-spectral evidence, correction scores, and deterministic inference code. Evidence-model and correction-score training code is not included.

The bundled pre-rank base responses are processed by the label-free prevalence estimator described in the paper. The corresponding reference prevalence values are included in `evidence/manifest.json` and checked during inference. Ground-truth change maps are not used by the inference pipeline; they are used only for final evaluation.

## Method Overview

```text
Base Counterfactual Evidence ----\
                                  > Scene-Level Evidence Routing
Boundary-Spectral Evidence -----/
                    |
                    v
Correction-Score Ranking
                    |
                    v
Selective Top-K Evidence Correction
                    |
                    v
Rank-Based Change Inference
```

## Installation

```bash
git clone https://github.com/RockAilab/CF-Evidence-HSICD.git
cd CF-Evidence-HSICD
python -m pip install -r requirements.txt
```

## Run All Scenes

```bash
python run_all.py
```

Results are written to `outputs/farmland/`, `outputs/hermiston/`, and `outputs/river/`.

## Run One Scene

```bash
python infer.py \
  --base-evidence evidence/river/base.npy \
  --boundary-evidence evidence/river/boundary.npy \
  --correction-score evidence/river/correction_score.npy \
  --correction-count 1615 \
  --estimated-prevalence 0.09206509952232868 \
  --output-dir outputs/river
```

The inputs correspond to the paper notation:

- `base-evidence`: base counterfactual evidence, `Eb`
- `boundary-evidence`: boundary-spectral evidence, `Es`
- `correction-score`: correction score, `p`
- `correction-count`: correction support size, `Kc`
- `estimated-prevalence`: label-free change prevalence, `pi_hat`

## Outputs

- `prediction.npy`: final binary change map, `Y_hat`
- `corrected_evidence.npy`: corrected evidence, `S`
- `alpha.npy`: evidence-reliance coefficient
- `correction_mask.npy`: selected correction support, `M_hat`
- `metadata.json`: scene-level inference metadata

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Supplementary Reproducibility Audit

The public audit regenerates the fixed-inference consistency tables and
sensitivity plots from the bundled evidence and read-only reference outputs.
It does not train a model, tune parameters, or modify the reported outputs.
The audit also regenerates the complete decision-level correction table,
including support, decision-change, repair/damage, and decision-margin
statistics.

```bash
python -m pip install -r requirements-supplementary.txt
python scripts/generate_supplementary.py
```

Generated audit files are written to `supplementary_outputs/`. The detailed
release-scope and cross-source record is provided in
`SUPPLEMENTARY_AUDIT.md`.

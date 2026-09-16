# HSI-CD Inference

[![Python](https://img.shields.io/badge/Python-%3E%3D3.9-3776AB.svg)](https://www.python.org/)
[![Scope](https://img.shields.io/badge/release-inference--only-2E7D32.svg)](#release-scope)
[![GT at inference](https://img.shields.io/badge/GT_at_inference-not_used-2E7D32.svg)](#inference-pipeline)

Inference-only implementation of the **retained-evidence sparse-correction v4**
decision stage for hyperspectral image change detection (HSI-CD).

The code performs prior-aware routing, sparse local correction, evidence
fusion, and label-free decision thresholding. It never accepts or reads a
ground-truth change map.

> **Important:** this repository begins at the post-training inference
> boundary. Retained evidence maps and learned correction outputs are inputs;
> their training code and experiment artifacts are intentionally not included.

## Inference Pipeline

```text
Base evidence ---------> stable percentile rank ----\
                                                       sparse fusion --> score S
Boundary evidence -----> stable percentile rank ----/
                                                        |
Correction probability --> Top-K correction mask ------/
Scene prior -----------> hard route + quantile threshold --> change map Y
```

Let `E_base` and `E_boundary` be the ranked evidence maps. The unlabeled scene
prior selects the initial route:

```text
route_alpha = 1,  if prior >= 0.25   (base route)
route_alpha = 0,  otherwise          (boundary route)
```

Given learned correction probability `p`, the `K_c` highest-probability pixels
form the sparse correction mask `M`. Alpha is the weight assigned to base
evidence. The correction endpoint is `0.95` for the base route and `0.20` for
the boundary route:

```text
M     = TopK(p, K_c)
alpha = route_alpha + (correction_alpha - route_alpha) * M
S     = alpha * E_base + (1 - alpha) * E_boundary
tau   = quantile(S, 1 - clip(prior, 0.01, 0.65))
Y     = S > tau
```

The stable rank transform, exact Top-K count, constants above, and strict `>`
decision match the original v4 inference protocol.

## Release Scope

Included:

- pure NumPy sparse-correction inference;
- command-line and Python APIs;
- validation and shape checks;
- synthetic unit tests.

Not included:

- training code, losses, optimizers, or pseudo-label construction;
- datasets or ground truth;
- retained evidence, learned alpha/probability maps, or checkpoints;
- experiment outputs, tables, figures, or benchmark metrics.

## Requirements

- Python 3.9 or newer
- NumPy 1.23 or newer

No GPU or deep-learning framework is required for this inference stage.

## Installation

```bash
git clone https://github.com/RockAilab/HSI-CD-Inference.git
cd HSI-CD-Inference
python -m pip install -e .
```

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

## Input Contract

All map inputs use NumPy `.npy` files and must have the same spatial shape
`[H, W]`.

| Input | Type | Description |
|---|---|---|
| `base_evidence` | 2D finite array | Retained base/CNN evidence; ranking is applied internally |
| `boundary_evidence` | 2D finite array | Retained boundary-spectral evidence; ranking is applied internally |
| `prior` | scalar in `[0, 1]` | Unlabeled scene-level change prior |
| `correction_probability` | 2D array in `[0, 1]` | Learned local correction probability |
| `correction_count` | integer | Unlabeled Top-K correction count `K_c` |
| `alpha_map` | 2D array in `[0, 1]` | Optional precomputed ranked alpha, used instead of probability and count |

Provide exactly one correction representation:

1. `correction_probability` together with `correction_count`; or
2. `alpha_map`.

The prior and correction count must be produced by the same fixed unlabeled
upstream protocol as the retained evidence. They must not be selected using GT
metrics.

## Command-Line Usage

### Correction probability

```bash
hsi-cd-infer \
  --base-evidence /path/to/base.npy \
  --boundary-evidence /path/to/boundary.npy \
  --correction-probability /path/to/correction_probability.npy \
  --correction-count 1615 \
  --prior 0.0920650995 \
  --output-dir outputs/example
```

### Precomputed alpha

```bash
hsi-cd-infer \
  --base-evidence /path/to/base.npy \
  --boundary-evidence /path/to/boundary.npy \
  --alpha-map /path/to/ranked_alpha.npy \
  --prior 0.0920650995 \
  --output-dir outputs/example
```

The repository-level wrapper exposes the same interface:

```bash
python infer.py --help
```

## Python API

```python
import numpy as np

from hsi_cd_inference import run_inference

base = np.load("/path/to/base.npy")
boundary = np.load("/path/to/boundary.npy")
probability = np.load("/path/to/correction_probability.npy")

result = run_inference(
    base,
    boundary,
    prior=0.0920650995,
    correction_probability=probability,
    correction_count=1615,
)

np.save("prediction.npy", result.prediction)
```

## Outputs

The CLI creates only the requested output directory:

| File | Description |
|---|---|
| `prediction.npy` | Binary `uint8` change map |
| `fused_score.npy` | Continuous fused evidence score `S` |
| `alpha.npy` | Ranked sparse base-branch weight map |
| `correction_mask.npy` | Pixels whose alpha differs from the hard route |
| `metadata.json` | Shape, prior, route, threshold, and pixel counts |

`metadata.json` records `ground_truth_used: false` for auditability.

## Reproduction Check

Before release, this implementation was replayed against the fixed formal v4
inputs for Farmland, Hermiston, and River. For all three scenes:

- generated alpha maps were exactly equal;
- fused scores were exactly equal (`max_abs_difference = 0`);
- final binary predictions were exactly equal (`mismatched_pixels = 0`).

This check used existing arrays outside the repository. No dataset, evidence,
GT, or result map is committed here.

## Repository Layout

```text
.
├── infer.py
├── pyproject.toml
├── src/hsi_cd_inference/
│   ├── __init__.py
│   ├── cli.py
│   └── pipeline.py
└── tests/
    └── test_pipeline.py
```

## Limitations

- This is the exact **inference/fusion stage**, not an end-to-end raw-HSI
  training package.
- The original sparse-gate experiments preserved probability and alpha maps,
  but did not preserve a reusable cross-scene gate checkpoint. This repository
  therefore does not claim checkpoint inference on unseen scenes.
- Generalization depends on the external retained evidence and correction
  outputs supplied by the user.
- Top-K selection assumes the supplied correction probabilities follow the
  calibration and counting protocol used by the original method.

## 中文说明

本仓库仅发布原始 retained-evidence sparse-correction v4 的**推理与融合部分**，
不包含训练代码、数据集、GT、checkpoint、中间 evidence/alpha 图或实验结果。
输入为已经得到的 base evidence、boundary evidence、correction probability（或
ranked alpha）及无标签 prior；输出为融合分数和二值变化图。该边界是有意设置的，
避免把仅能复现后端决策的代码误述为可从原始 HSI 重新训练完整模型的仓库。

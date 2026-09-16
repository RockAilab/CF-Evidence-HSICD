# HSI-CD Inference

Inference-only implementation of the retained-evidence sparse-correction v4
decision stage for hyperspectral image change detection.

This repository intentionally contains no training code, datasets, ground
truth, trained experiment directories, intermediate maps, or benchmark result
tables. It starts at the post-training inference boundary and consumes:

1. a retained base evidence map;
2. a retained boundary-spectral evidence map;
3. either a learned correction-probability map plus its unlabeled correction
   count, or an already ranked alpha map; and
4. the unlabeled scene change prior.

It produces the ranked sparse alpha map, fused score, final binary change map,
and an inference metadata JSON file. Ground truth is neither accepted nor read.

## Method

The two evidence maps are first converted to stable percentile ranks. Let
`E_base` and `E_boundary` denote those ranked maps. The scene prior selects the
hard route:

```text
route_alpha = 1, if prior >= 0.25
route_alpha = 0, otherwise
```

For learned correction probability `p`, the top `K_c` pixels form the sparse
correction mask. The correction endpoint is `0.95` for the dense route and
`0.20` for the boundary route:

```text
alpha = route_alpha + (correction_alpha - route_alpha) * TopK(p, K_c)
S = alpha * E_base + (1 - alpha) * E_boundary
tau = quantile(S, 1 - prior)
Y = S > tau
```

These constants and the strict `>` decision reproduce the original v4
inference protocol.

## Installation

Python 3.9 or newer is recommended.

```bash
python -m pip install -e .
```

Run the dependency-free unit tests:

```bash
python -m unittest discover -s tests -v
```

## Usage

Using a learned correction-probability map:

```bash
hsi-cd-infer \
  --base-evidence /path/to/base.npy \
  --boundary-evidence /path/to/boundary.npy \
  --correction-probability /path/to/correction_probability.npy \
  --correction-count 1615 \
  --prior 0.0920650995 \
  --output-dir outputs/river
```

Using an exported ranked alpha map:

```bash
hsi-cd-infer \
  --base-evidence /path/to/base.npy \
  --boundary-evidence /path/to/boundary.npy \
  --alpha-map /path/to/ranked_alpha.npy \
  --prior 0.0920650995 \
  --output-dir outputs/river
```

The repository-level `infer.py` wrapper provides the same command:

```bash
python infer.py --help
```

## Outputs

The output directory contains:

- `prediction.npy`: binary `uint8` change map;
- `fused_score.npy`: continuous fused evidence score;
- `alpha.npy`: sparse base-branch alpha map;
- `correction_mask.npy`: pixels whose alpha differs from the hard route;
- `metadata.json`: shapes, prior, threshold, routing, and pixel counts.

## Scope And Limitations

- This is the exact **inference/fusion stage**, not a raw-HSI-to-map training
  package.
- Retained evidence and the learned correction output are external inputs. They
  are not bundled because this release excludes training code and experiment
  artifacts.
- The original sparse-gate runs preserved probability/alpha maps rather than a
  reusable cross-scene gate checkpoint. Accordingly, this release does not
  claim checkpoint inference on a previously unseen scene.
- The unlabeled prior and correction count must come from the same fixed
  upstream protocol used to produce the evidence package. They must not be
  selected from ground-truth metrics.

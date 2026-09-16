# Label-Free Counterfactual Evidence Inference with Anchor-Induced Sparse Correction for Hyperspectral Change Detection

Inference code for the retained-evidence sparse-correction model described in
the paper.

## Installation

```bash
git clone https://github.com/RockAilab/Label-Free-Counterfactual-Evidence-Inference-with-Anchor-Induced-Sparse-Correction-for-HSI-CD.git
cd Label-Free-Counterfactual-Evidence-Inference-with-Anchor-Induced-Sparse-Correction-for-HSI-CD
python -m pip install -e .
```

## Prepare Inputs

Prepare the following two-dimensional NumPy arrays with the same spatial shape:

- base evidence: `base.npy`
- boundary-spectral evidence: `boundary.npy`
- learned correction probability: `correction_probability.npy`

You also need the scene prior and correction count produced by the upstream
pipeline. If a ranked alpha map is already available, it can be used instead of
the correction probability and correction count.

## Run Inference

### Using correction probability

```bash
export SCENE_PRIOR=...
export CORRECTION_COUNT=...

hsi-cd-infer \
  --base-evidence /path/to/base.npy \
  --boundary-evidence /path/to/boundary.npy \
  --correction-probability /path/to/correction_probability.npy \
  --correction-count "$CORRECTION_COUNT" \
  --prior "$SCENE_PRIOR" \
  --output-dir outputs/example
```

### Using a ranked alpha map

```bash
export SCENE_PRIOR=...

hsi-cd-infer \
  --base-evidence /path/to/base.npy \
  --boundary-evidence /path/to/boundary.npy \
  --alpha-map /path/to/ranked_alpha.npy \
  --prior "$SCENE_PRIOR" \
  --output-dir outputs/example
```

The repository-level entry point provides the same interface:

```bash
python infer.py --help
```

## Outputs

Inference writes the following files to `--output-dir`:

- `prediction.npy`: binary change map
- `fused_score.npy`: fused evidence score
- `alpha.npy`: sparse fusion alpha map
- `correction_mask.npy`: selected correction support
- `metadata.json`: inference metadata

## Verify Installation

```bash
python -m unittest discover -s tests -v
```

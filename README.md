# Label-Free Counterfactual Evidence Inference with Anchor-Induced Sparse Correction for Hyperspectral Change Detection

This repository contains the inference code, benchmark data, and evidence required to reproduce the change maps for Farmland, Hermiston, and River.

## Installation

```bash
git clone https://github.com/RockAilab/Label-Free-Counterfactual-Evidence-Inference-with-Anchor-Induced-Sparse-Correction-for-HSI-CD.git
cd Label-Free-Counterfactual-Evidence-Inference-with-Anchor-Induced-Sparse-Correction-for-HSI-CD
python -m pip install -r requirements.txt
```

## Run Inference

The datasets and inference evidence are included in `data/` and `evidence/`. Run all three scenes with:

```bash
python run_all.py
```

Results are written to:

```text
outputs/farmland/
outputs/hermiston/
outputs/river/
```

Each directory contains the binary change map (`prediction.npy`), fused score, fusion map, correction mask, and inference metadata.

To run a single scene through the command-line interface:

```bash
python infer.py \
  --base-evidence evidence/river/base.npy \
  --boundary-evidence evidence/river/boundary.npy \
  --correction-probability evidence/river/correction_probability.npy \
  --correction-count 1615 \
  --prior 0.09206509952232868 \
  --output-dir outputs/river
```

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

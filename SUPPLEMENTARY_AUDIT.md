# Supplementary Reproducibility Audit

## Release Version

- Paper release tag: `paper-submission-v1`
- Commit: `aac7f1fc9dfc4cfe00f5b2e7045cfb16b383b66d`

## Public Release Scope

This repository releases benchmark data, fixed evidence fields, correction
scores, read-only reference outputs, and deterministic inference code. It
supports fixed inference reproduction. Evidence-model and correction-score
training code is not included.

Ground-truth maps are not accessed by prevalence estimation, routing,
selective correction, or final inference. The supplementary script first
reconstructs and verifies all fixed-inference outputs. It loads ground truth
only afterward for post-hoc metrics and diagnostic visualization.

The historical Farmland upstream initialization cannot currently be
reconstructed from the original image pair. Farmland therefore has the
following explicit boundary:

```text
Farmland historical initialization = Fixed-input reproduction only
```

All downstream prevalence estimation, routing, correction-mask construction,
corrected-evidence computation, and final inference are implemented by the
released code.

## Entry Points

| Purpose | Path |
|---|---|
| Run all fixed-inference scenes | `run_all.py` |
| Run one scene | `infer.py` |
| Generate supplementary audit | `scripts/generate_supplementary.py` |
| Fixed inference implementation | `src/hsi_cd_inference/pipeline.py` |
| Prevalence and routing implementation | `src/hsi_cd_inference/prevalence.py` |

## Cross-Source Record

Running `python scripts/generate_supplementary.py` writes a machine-readable
record to `supplementary_outputs/tables/reproducibility_audit.csv` with these
columns:

```text
Dataset
Data source
Code entry point
GT usage
Parameter source
Output file
Check
Expected value
Observed value
Mismatch
Status
```

The audit covers Farmland, Hermiston, and River for:

| Check | Expected result | GT usage |
|---|---|---|
| Prevalence | Manifest value reproduced | No |
| Scene-level routing | Reference route reproduced | No |
| Correction score | Bundled input hash reproduced | No |
| Correction mask | Reference mask reproduced exactly | No |
| Corrected evidence | Reference field reproduced within `1e-7` | No |
| Final prediction | Reference map reproduced exactly | No |
| Decision-level correction audit | Table S4 fully regenerated | GT post-hoc only |
| Farmland historical initialization | Fixed-input reproduction only | No |

Any mismatch terminates the script before post-hoc evaluation products are
generated. The script never performs parameter selection or modifies bundled
evidence, correction scores, or reference outputs.

After fixed predictions are verified, the script loads GT for post-hoc
classification of decision changes. It regenerates Table S4 fields for support,
decision change, in-support flip rate, repair, damage, repair/damage ratio,
repair margin, and damage margin. Every field is checked against the fixed
reported result, and any mismatch terminates the audit.

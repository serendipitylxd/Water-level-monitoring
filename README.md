# LiDAR Water-Level Monitoring in Navigation Locks

This repository contains the code and reproducibility artifacts for **LiDAR-Based Noncontact Water-Level Sensing in Navigation Locks via Wall-Belt Geometry and Temporal Filtering**. The pipeline removes ships and lock infrastructure with a frozen six-detector ensemble, constructs wall-belt height profiles, estimates water level with regression models, and optionally applies causal stabilization and Kalman filtering.

The current release uses operation-disjoint evaluation. The earlier frame-random artifacts under `outputs/wl_*` are retained only for historical compatibility and are not the results reported below.

## Release history

| Release | Release date |
|---|---|
| v0.1.0 | 2025-09-26 |
| v0.2.0 | 2026-05-22 |
| v0.3.0 | 2026-08-28 |

## What changed in v0.3.0

- Added the exact 8,000-frame water-level manifest with 18 lock operations.
- Added the fixed operation split: operations 1–10 for training, 11–13 for validation, and 14–18 for testing.
- Added per-operation results for all 13 estimators.
- Added 18-fold leave-one-operation-out evaluation for Linear Regression, MLP, Random Forest, and HGRN.
- Added paired primary-output versus Kalman-output lag analysis.
- Added staff-gauge reference uncertainty and measured end-to-end runtime artifacts.
- Updated the training, evaluation, Random Forest, and feature-cache code used by the revised protocol.
- Replaced machine-specific paths with repository-relative paths or public placeholders.

## Evaluation protocol

The detector corpus contains 16,000 annotated frames. The water-level study uses only the 8,000-frame detector-test shard, so detector-development frames are excluded from water-level model fitting and evaluation. These 8,000 frames cover 18 complete lock operations collected over three days.

| Role | Operations | Frames | Dates |
|---|---:|---:|---|
| Training | 1–10 | 4,845 | 2024-07-17 to 2024-07-18 |
| Validation | 11–13 | 977 | 2024-07-18 |
| Test | 14–18 | 2,178 | 2024-07-19 |

No operation appears in more than one role, and the test operations are also day-disjoint from the development operations. The native LiDAR rate is 10 Hz; the released water-level shard retains approximately one timestamp-selected frame every 2 s, with longer gaps where scans are missing.

The public manifest is [`splits/waterlevel_test_only_8000/manifest.csv`](splits/waterlevel_test_only_8000/manifest.csv), and its split audit is [`audit.json`](splits/waterlevel_test_only_8000/audit.json).

## Fixed-test results

The table reports pooled metrics on operations 14–18. Error metrics are in centimetres; correlation is in percent. “KF RMSE” is computed after the model's primary output. The Random Forest primary output includes its validation-selected causal first-order stabilizer.

| Model | MAE | RMSE | Bias | Corr | KF RMSE |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 31.76 | 42.02 | -5.43 | 28.78 | 33.12 |
| Ridge Regression | 30.21 | 38.48 | -4.29 | 30.01 | 31.65 |
| SVR | 6.31 | 7.84 | +5.12 | 12.21 | 7.37 |
| Random Forest | **2.19** | **2.95** | **+1.50** | **68.70** | **2.95** |
| MLP | 4.67 | 6.76 | +2.33 | 27.98 | 6.39 |
| 1D-CNN | 5.31 | 6.66 | +5.03 | 31.03 | 6.48 |
| HGRN | 5.40 | 6.67 | +5.34 | 28.96 | 6.66 |
| Hyena | 6.15 | 7.49 | +6.07 | 30.20 | 7.49 |
| Mamba | 5.36 | 6.72 | +5.18 | 22.07 | 6.72 |
| MEGA | 9.38 | 10.51 | +9.38 | 20.59 | 10.49 |
| RetNet | 6.53 | 7.96 | +6.33 | 31.77 | 7.95 |
| RWKV | 8.26 | 9.35 | +8.26 | 11.60 | 9.35 |
| Transformer | 6.96 | 8.43 | +6.87 | 24.31 | 8.42 |

Exact floating-point values and all 65 model-operation rows are available in [`results/operation_split`](results/operation_split/README.md).

## Leave-one-operation-out evaluation

Each fold holds out one entire operation and fits on the other 17. Hyperparameters and epoch counts remain fixed across folds. Values below are operation-macro mean ± sample standard deviation in centimetres.

| Model | Primary MAE | Primary RMSE | KF MAE | KF RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 26.74 ± 9.24 | 36.72 ± 13.48 | 22.45 ± 9.80 | 30.37 ± 13.80 |
| MLP | 5.17 ± 2.89 | 6.85 ± 3.85 | 5.02 ± 2.86 | 6.47 ± 3.69 |
| Random Forest | **1.53 ± 0.45** | **2.19 ± 0.64** | **1.53 ± 0.45** | **2.19 ± 0.64** |
| HGRN | 2.79 ± 1.26 | 3.54 ± 1.36 | 2.79 ± 1.27 | 3.56 ± 1.38 |

## Repository layout

```text
configs/operation_split/              current 13-model operation-split configs
outputs/det3d/                        frozen per-detector and fused detections
pcdet_models/config/                  OpenPCDet model configs
results/operation_split/              lightweight tables and protocol audits
scripts/                              training, evaluation, LOOCV, lag, and runtime tools
splits/waterlevel_test_only_8000/     public 8,000-frame manifest and split audit
trout_add_info/                       legacy timestamp/water-level metadata
utils/                                data, model, metric, and filtering utilities
```

## Installation

Python 3.8 and the CUDA 11.8 build of PyTorch 2.1.0 were used for the reported runs.

```bash
conda create -n trout python=3.8
conda activate trout
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

OpenPCDet is required only to rerun the six 3D detectors. The repository already contains their fused detection JSON for the released shard. Detector setup and checkpoints follow [Hydro3DNet](https://github.com/serendipitylxd/Hydro3DNet).

## Data and portable paths

Download the TROUT data archive from [Google Drive](https://drive.google.com/file/d/1E5cQHYgv8s7pCfQyvnH2FvPhxDPe2lxb/view?usp=sharing), then set the dataset root. Keep the placeholder style in committed files; do not commit a local absolute path.

```bash
export TROUT_ROOT=path/to/TROUT
```

The manifest expands `${TROUT_ROOT}` at runtime. As an alternative, generate a git-ignored local manifest:

```bash
python scripts/relocate_manifest.py \
  --points-dir path/to/TROUT/points_test \
  --check-files
```

If using `manifest.local.csv`, change `data.operation_manifest_path` only in your uncommitted local config.

Run the public-path guard before committing changes:

```bash
python scripts/check_public_paths.py
```

## Reproduce the water-level experiments

### 1. Optional: rerun the detector ensemble

Edit only the public placeholders in `configs/infer_3d_pcdet.yaml`, then run:

```bash
python scripts/infer_3d_pcdet.py \
  --cfg configs/infer_3d_pcdet.yaml \
  --pcdet_tools path/to/OpenPCDet/tools
```

Fusion uses a score threshold of 0.40, BEV IoU threshold of 0.70, and vote ratio of 0.50, requiring agreement from at least three of six detectors.

### 2. Build the deterministic wall-profile cache

```bash
python scripts/build_manifest_feature_cache.py \
  --cfg configs/operation_split/wl_hgrn.yaml
```

The cache contains features and masks, not target labels. Its manifest and preprocessing hashes are checked when loaded.

### 3. Run the fixed 13-model benchmark

```bash
python scripts/run_operation_benchmark.py \
  --manifest splits/waterlevel_test_only_8000/manifest.csv \
  --continue-on-error
```

The classical and neural estimators share the same samples, geometry, feature cache, and semantic split. Model outputs are written under `outputs_operation_split/test_only_8000/fixed/`.

### 4. Run the Random Forest primary pipeline

All selection below uses development operations only. The selected model is refit on operations 1–13 before evaluating operations 14–18.

```bash
python scripts/tune_anchored_random_forest_operation_split.py tune \
  --coefficient-objective robust_average_rmse
python scripts/tune_anchored_random_forest_operation_split.py refit
python scripts/tune_anchored_random_forest_operation_split.py evaluate
python scripts/compose_rf_manuscript_reporting.py
```

### 5. Run four-model LOOCV

```bash
python scripts/run_operation_loocv.py --cfg configs/operation_split/wl_linear_regression.yaml --epochs 1 --run
python scripts/run_operation_loocv.py --cfg configs/operation_split/wl_mlp.yaml --epochs 22 --run
python scripts/run_operation_loocv.py --cfg configs/operation_split/wl_hgrn.yaml --epochs 36 --run
python scripts/run_anchored_rf_loocv.py \
  --frozen outputs_operation_split/test_only_8000/rf_tuned_robust_anchor/frozen_selection.json
python scripts/summarize_loocv_benchmark.py
```

### 6. Reference and timing audits

```bash
python scripts/audit_test_only_reference_protocol.py
python scripts/benchmark_full_frozen_pipeline.py \
  --pcdet-tools path/to/OpenPCDet/tools
```

The measured full pipeline averaged 561.04 ± 46.96 ms per frame on one RTX 4080 (100 measured frames after warm-up), corresponding to 1.78 frames/s. Stage-level values are included in the result artifacts.

## Detector accuracy

| Detector | Ship mAP\_3D (test) | Infra mAP\_3D (test) |
|---|---:|---:|
| PointPillars | 61.6 | 99.4 |
| SECOND | 51.6 | 98.6 |
| DSVT-Voxel | 54.7 | 95.5 |
| Voxel-Mamba | 57.8 | 96.2 |
| PV-RCNN | 88.1 | 99.8 |
| PV-RCNN++ | 88.5 | 99.9 |

## License

This project is released under the [Apache License 2.0](LICENSE).

## Citation

```bibtex
@software{lu2026waterlevelmonitoring,
  author  = {Xiaodong Lu},
  title   = {LiDAR-Based Noncontact Water-Level Sensing in Navigation Locks via Wall-Belt Geometry and Temporal Filtering},
  year    = {2026},
  url     = {https://github.com/serendipitylxd/Water-level-monitoring}
}
```

# Operation-split result artifacts

These lightweight files correspond to the current 8,000-frame, 18-operation water-level protocol. Unless a filename or column states otherwise, water-level errors in CSV files are stored in metres and correlations are unitless. The rounded centimetre values in the main README are derived from these files.

| Artifact | Contents |
|---|---|
| `fixed_test_overall.csv` | Pooled and operation-macro results for all 13 models on operations 14–18 |
| `fixed_test_per_operation.csv` | One row per model and held-out test operation (65 rows) |
| `fixed_test_audit.json` | Split, source-definition, hash, and row-count checks |
| `random_forest_selection.json` | Frozen RF representation, parameters, causal coefficient, and Kalman settings |
| `loocv_macro.csv` | Mean and sample standard deviation across 18 held-out-operation folds |
| `loocv_per_operation.csv` | All four-model fold metrics (72 rows) |
| `loocv_kf_rmse_by_operation.csv` | Compact KF RMSE view by model and operation |
| `loocv_audit.json` | LOOCV completeness and operation-disjointness audit |
| `paired_kf_lag_summary.csv` | Primary-output and KF-output lag summaries across test operations |
| `paired_kf_lag_by_operation.csv` | Paired lag evidence for each model-operation pair |
| `paired_kf_lag_audit.json` | Lag definition, cadence, and arithmetic audit |
| `reference_protocol.json` | Gauge reading, synchronization, spatial relation, and uncertainty protocol |
| `reference_uncertainty_components.csv` | Quantified components of the ±0.52 cm operational reference bound |
| `reference_cadence_by_operation.csv` | Timestamp cadence and maximum observed water-level rate by operation |
| `detector_accuracy.csv` | Ship and infrastructure test mAP\_3D for each frozen detector |
| `runtime_stages.csv` | Measured stage-wise latency in milliseconds |
| `runtime_summary.json` | Hardware, scope, sample count, end-to-end latency, and throughput |

The evaluation-manifest hash in the audits identifies the original experiment manifest. The public manifest has the same sample identities, targets, timestamps, operation IDs, and split roles; only its machine-specific point and annotation paths were replaced with portable placeholders.

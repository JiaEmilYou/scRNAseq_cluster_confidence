# Dev Notes

## Contents
- [MLflow + DVC interaction](#mlflow--dvc-interaction)
- [MLflow setup](#mlflow-setup)
- [DVC experiments](#dvc-experiments)
- [Batch experiments](#batch-experiments)
- [Output overwriting](#output-overwriting)
- [Scanpy preprocessing](#scanpy-preprocessing-logic)
- [LightGBM interpretation](#lightgbm-labels-are-derived-from-leiden)
- [Confidence metrics](#confidence-metrics)
- [Common errors](#common-errors)
- [Design choices](#design-choices)


## MLflow + DVC interaction

- MLflow DB stored outside repo: ../mlflow_dbs/
- Reason:
  - DVC may try to remove/overwrite files in repo
  - causes file locking errors with SQLite
- Always:
  - start MLflow server before running experiments
  - run experiments in a separate terminal


 ## MLflow setup

- Start server:
  mlflow ui --backend-store-uri sqlite:///../mlflow_dbs/scRNAseq_lightgbm.db

- Default UI:
  http://127.0.0.1:5000

- In code:
  use `MLFLOW_TRACKING_URI` with fallback to `http://127.0.0.1:5000`

- Behavior:
  - If `MLFLOW_TRACKING_URI` is **not set**:
    → pipeline uses default `http://127.0.0.1:5000`
  - If `MLFLOW_TRACKING_URI` **is set**:
    → pipeline sends results to that address instead

- Example (custom port or remote server):
  export MLFLOW_TRACKING_URI=http://127.0.0.1:6000

- Important:
  `mlflow ui` does not automatically set `MLFLOW_TRACKING_URI`



  ## DVC experiments

- Run:
  dvc exp run -n <name> -S key=value

- Overwrite same name:
  dvc exp run -f -n <name>

- List experiments:
  dvc exp show

- Experiments stored in:
  .dvc/tmp/exps/


  ## Batch experiments

- Loop over datasets + resolutions:

datasets = pbmc3k, pbmc68k_reduced, paul15
resolutions = 0.5, 1.0, 1.5, 2.0, 2.5

- Use bash loop (run_experiments.sh)
bash run_experiments.sh

- Important:
  unique naming required OR use -f


## Output overwriting

- DVC outputs (artifacts/) are overwritten each run
- Figures are copied to results/ for persistent storage

Example:
artifacts/figures/pbmc3k/res_1p0/   → latest run
results/figures/pbmc3k/res_1p0/     → saved outputs

Note:
- rerunning same dataset + resolution overwrites results unless timestamped

## Scanpy preprocessing logic

- Raw datasets (pbmc3k, paul15):
  - normalize_total
  - log1p
  - HVG selection
  - PCA

- pbmc68k_reduced:
  - already preprocessed
  - only run PCA if missing

Note:
- pbmc3k and paul15 use identical preprocessing
- differences arise from underlying biological structure (discrete vs continuous)


  ## LightGBM labels are derived from Leiden

- Model is trained on Leiden clusters
- Not independent ground truth

Interpretation:
- high confidence → well-separated clusters
- low confidence → ambiguous / continuous structure

Use:
- evaluate clustering stability, not classification accuracy alone


## Confidence metrics

### Per-cell metrics
Stored in `adata.obs`:
- `confidence_lgbm` → max predicted probability
- `entropy_lgbm` → entropy of class probabilities
- `entropy_norm_lgbm` → entropy normalized by number of classes
- `margin_lgbm` → difference between top two probabilities

Low confidence defined by:
  max(probability) < threshold (default 0.6)

---

### Dataset-level summary metrics
Computed over all cells:
- `mean_entropy_norm_all_cells`
- `p95_entropy_norm_all_cells`
- `p99_entropy_norm_all_cells`
- `mean_margin_all_cells`
- `p05_margin_all_cells`

These capture:
- overall uncertainty (mean)
- tail ambiguity (p95 / p99)
- separability between clusters (margin)

---

### Cross-validation stability (NEW)

Repeated StratifiedKFold is used to evaluate robustness:

- `cv_mean_entropy_norm_mean / std`
- `cv_mean_margin_mean / std`
- `cv_accuracy_mean / std`
- `cv_macro_f1_mean / std`

Interpretation:
- low std → stable clustering
- high std → unstable / over-partitioned clustering

---

### Interpretation

The goal is not classification accuracy alone.

Instead:
- high confidence + low entropy → well-separated clusters
- increasing entropy + decreasing margin → over-clustering
- smooth degradation across resolution → continuous structure
- sharp degradation → discrete clusters being split


## Common errors

- HVG failure (NaN bins):
  → caused by zero-count cells

- MLflow error:
  → tracking URI mismatch (sqlite vs http)

- File locking:
  → MLflow DB inside repo


  ## Design choices

- DVC:
  reproducibility (data + pipeline)

- MLflow:
  experiment tracking (metrics + params)

- LightGBM:
  interpretable confidence on clustering

- Goal:
  quantify clustering reliability, not replace clustering
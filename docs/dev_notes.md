# Dev Notes

## Contents

- [MLflow and DVC interaction](#mlflow-and-dvc-interaction)
- [MLflow setup](#mlflow-setup)
- [DVC experiments](#dvc-experiments)
- [Batch experiments](#batch-experiments)
- [Custom datasets](#custom-datasets)
- [Output overwriting and archives](#output-overwriting-and-archives)
- [Scanpy preprocessing logic](#scanpy-preprocessing-logic)
- [LightGBM interpretation](#lightgbm-interpretation)
- [Confidence metrics](#confidence-metrics)
- [Wrong-or-tail diagnostics](#wrong-or-tail-diagnostics)
- [Common errors](#common-errors)
- [Design choices](#design-choices)

## MLflow and DVC interaction

- MLflow should be started before running DVC experiments.
- Run MLflow in a separate terminal.
- Keep the MLflow SQLite database outside the repo, for example:

```text
../mlflow_dbs/scRNAseq_lightgbm.db
```

Reason:

- DVC may remove or overwrite tracked outputs during experiments.
- Keeping the MLflow DB inside the repo can cause file-locking problems, especially on Windows.

## MLflow setup

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///../mlflow_dbs/scRNAseq_lightgbm.db
```

Default UI:

```text
http://127.0.0.1:5000
```

The LightGBM pipeline uses:

```text
MLFLOW_TRACKING_URI
```

with fallback:

```text
http://127.0.0.1:5000
```

If using another port:

```bash
mlflow ui --backend-store-uri sqlite:///../mlflow_dbs/scRNAseq_lightgbm.db --port 6000
export MLFLOW_TRACKING_URI=http://127.0.0.1:6000
```

Important:

- `mlflow ui` does not automatically set `MLFLOW_TRACKING_URI`.
- The pipeline will fail with a connection refused error if it tries to log to an MLflow server that is not running.

## DVC experiments

Run one experiment:

```bash
dvc exp run -n <name> -S key=value
```

Overwrite an experiment with the same name:

```bash
dvc exp run -f -n <name> -S key=value
```

List experiments:

```bash
dvc exp show
```

Experiments are stored under:

```text
.dvc/tmp/exps/
```

## Batch experiments

The standard batch grid uses:

```text
datasets = pbmc3k, pbmc68k_reduced, paul15
resolutions = 0.5, 1.0, 1.5, 2.0, 2.5
```

Run:

```bash
bash run_experiments.sh
```

Use unique experiment names, or use `-f` to overwrite existing names.

## Custom datasets

Custom datasets can be passed through:

```yaml
scanpy:
  input_path: data/raw/my_dataset.h5ad
  dataset_name: my_dataset
```

Example:

```bash
dvc exp run -f -n my_dataset_res_0p5 \
  -S scanpy.input_path=data/raw/my_dataset.h5ad \
  -S scanpy.dataset_name=my_dataset \
  -S scanpy.leiden_resolution=0.5
```

Current assumption:

- Custom inputs are raw-count-like `.h5ad` files.
- The pipeline will normalize, log-transform, select HVGs, compute PCA, neighbors, Leiden, and UMAP.

Caveat:

- If a custom dataset is already normalized/logged/preprocessed, the current pipeline may preprocess it again.
- A future option such as `scanpy.preprocessing_mode: raw/preprocessed` would make this explicit.

DVC caveat:

- `scanpy.input_path` is tracked as a parameter.
- The actual custom input file is not currently listed as a DVC dependency.
- If the file contents change but the path stays the same, DVC may not automatically detect the change.

## Output overwriting and archives

DVC outputs are overwritten each run:

```text
data/processed/processed.h5ad
data/processed/annotated.h5ad
artifacts/models/lightgbm_model.txt
artifacts/metrics/metrics.json
artifacts/figures/
```

Archived copies are saved separately:

```text
data/processed/archive/<dataset>_res_<resolution>.h5ad
artifacts/metrics/archive/<dataset>_res_<resolution>.json
results/figures/<dataset>/res_<resolution>/
```

Example:

```text
artifacts/figures/pbmc3k/res_1p0/  = latest DVC output
results/figures/pbmc3k/res_1p0/    = archived figure copy
```

Note:

- Rerunning the same dataset and resolution overwrites archived outputs unless filenames are timestamped.

## Scanpy preprocessing logic

Raw-style datasets:

- custom `input_path`
- `pbmc3k`
- `paul15`

Workflow:

- `normalize_total`
- `log1p`
- highly variable gene selection
- PCA
- neighbors
- Leiden
- UMAP

`pbmc68k_reduced`:

- treated as preprocessed
- PCA is computed only if `X_pca` is missing
- neighbors, Leiden, and UMAP are recomputed

## LightGBM interpretation

LightGBM is trained to recover Leiden cluster labels from PCA features.

The labels are therefore not independent biological ground truth.

Interpretation:

- high confidence means the Leiden assignment is easy to recover from PCA space
- low confidence means the cell is ambiguous for the supervised model
- disagreement between Leiden and LightGBM suggests graph clustering and feature-space classification disagree for that cell

Use:

- evaluate clustering separability
- identify ambiguous cells
- compare behavior across Leiden resolutions
- diagnose possible over-clustering

## Confidence metrics

### Per-cell metrics

Stored in `adata.obs`:

```text
predicted_cluster_lgbm
confidence_lgbm
entropy_lgbm
entropy_norm_lgbm
margin_lgbm
prediction_matches_leiden
low_confidence_lgbm
oof_fold_lgbm
```

Stored in `adata.obsm`:

```text
probs_lgbm
```

Stored in `adata.uns`:

```text
probs_lgbm_columns
probs_lgbm_source
```

Current per-cell probabilities are out-of-fold:

```text
adata.uns["probs_lgbm_source"] = "out_of_fold"
```

This means each cell's saved LightGBM probabilities come from a model that did not train on that cell.

### Metric definitions

`confidence_lgbm`:

- maximum predicted probability

`entropy_lgbm`:

- entropy of the full predicted probability distribution

`entropy_norm_lgbm`:

- entropy normalized by the number of clusters

`margin_lgbm`:

- top predicted probability minus second-highest predicted probability

`low_confidence_lgbm`:

- `confidence_lgbm < low_confidence_threshold`
- default threshold: `0.6`

### Dataset-level summary metrics

Examples:

```text
mean_entropy_norm_all_cells
p95_entropy_norm_all_cells
p99_entropy_norm_all_cells
mean_margin_all_cells
p05_margin_all_cells
mean_confidence_all_cells
low_confidence_fraction_all_cells
```

These summarize overall uncertainty, tail ambiguity, and cluster separability.

### Cross-validation stability

Repeated `StratifiedKFold` is used for robustness summaries:

```text
cv_accuracy_mean / std
cv_macro_f1_mean / std
cv_mean_confidence_test_mean / std
cv_low_confidence_fraction_test_mean / std
cv_mean_entropy_norm_mean / std
cv_mean_margin_mean / std
```

Interpretation:

- low standard deviation means stable behavior across splits
- high standard deviation suggests unstable or split-dependent clustering

### Final model

After OOF metrics are computed, the saved final LightGBM model is trained on all cells:

```text
final_model.fit(X, y)
```

The final model is for reuse. Per-cell diagnostic metrics should be interpreted from OOF predictions, not from in-sample predictions.

## Wrong-or-tail diagnostics

The visualization stage now includes:

```text
umap_wrong_or_tail.png
```

This plot highlights cells that are either:

- wrong prediction: `predicted_cluster_lgbm != leiden`
- in any uncertainty tail:
  - low margin
  - high normalized entropy
  - low confidence

Current tail size:

```text
tail_fraction = min(wrong_fraction * 2, 0.5)
```

Interpretation:

- 1x tail means flag the same fraction of cells as the wrong-prediction rate
- 2x tail means flag twice the wrong-prediction rate

If 5% of cells are misclassified:

- 1x tail = most uncertain 5% of cells
- 2x tail = most uncertain 10% of cells

Observed pattern in archived diagnostics:

- 1x tails capture about half of wrong predictions
- 2x tails capture roughly three quarters of wrong predictions
- confidence, entropy, and margin tails are highly consistent with each other

## Common errors

### MLflow connection refused

Typical cause:

- MLflow server is not running at `http://127.0.0.1:5000`

Fix:

```bash
mlflow ui --backend-store-uri sqlite:///../mlflow_dbs/scRNAseq_lightgbm.db
```

### HVG failure or NaN bins

Possible causes:

- zero-count cells
- already transformed data being treated as raw counts

Potential fixes:

- filter zero-count cells before preprocessing
- add an explicit preprocessing mode for preprocessed inputs

### Small Leiden clusters

LightGBM CV requires:

```text
smallest cluster size >= cv_folds
```

If Leiden creates tiny clusters, reduce `cv_folds` or lower Leiden resolution.

### DVC output confusion

`artifacts/figures/` contains only the latest DVC output.

Historical plots are under:

```text
results/figures/<dataset>/res_<resolution>/
```

On Windows, check "Date modified" rather than "Date created" when verifying updates.

## Design choices

DVC:

- reproducible data and pipeline versioning

MLflow:

- experiment tracking for parameters, metrics, and artifacts

Scanpy + Leiden:

- standard unsupervised scRNA-seq clustering workflow

LightGBM:

- supervised cluster recoverability analysis
- per-cell confidence and uncertainty diagnostics

Main goal:

- quantify clustering reliability and ambiguity, not replace clustering

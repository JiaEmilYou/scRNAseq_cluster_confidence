This project implements a reproducible pipeline for single-cell RNA-seq (scRNA-seq) analysis and classification.

In addition to standard clustering (Scanpy + Leiden), it introduces a supervised learning step (LightGBM) to assign cluster labels and provide **confidence and uncertainty metrics for each cell**.

Rather than relying only on predicted probabilities, the pipeline computes multiple complementary measures, including:
- prediction confidence (max probability)
- entropy (uncertainty of class distribution)
- margin (difference between top predictions)

These metrics are further summarized at the dataset level and evaluated across **repeated cross-validation splits**, allowing assessment of:
- how reliably clusters can be separated
- how confidence changes across clustering resolutions
- whether structure is discrete or continuous

The pipeline is designed for reproducibility and experimentation, using DVC for data and pipeline versioning, and MLflow for tracking model performance, uncertainty metrics, and experiment configurations across datasets and clustering parameters.

## Pipeline overview

The pipeline has three main DVC stages:

1. **Scanpy preprocessing**
   - load an `.h5ad` dataset
   - normalize counts
   - log-transform expression values
   - select highly variable genes
   - compute PCA
   - build a neighbor graph
   - run Leiden clustering
   - compute UMAP

2. **LightGBM confidence analysis**
   - train LightGBM to recover Leiden clusters from PCA features
   - evaluate cluster recoverability with cross-validation
   - save per-cell confidence, entropy, margin, and prediction agreement metrics

3. **Visualization**
   - generate UMAP plots for Leiden clusters, LightGBM predictions, confidence, agreement, and wrong/uncertain cells
   - archive figures by dataset and Leiden resolution

## Running a new dataset

The pipeline can be run on a custom single-cell RNA-seq dataset stored as an `.h5ad` file. The expected input is a raw-count-like `AnnData` object that can be processed by the standard Scanpy workflow in this project.

The input `.h5ad` should contain cells in rows and genes/features in columns:

```text
adata.X   = raw count matrix
adata.obs = cell metadata
adata.var = gene metadata
```

Place the input file somewhere inside the project, for example:

```text
data/raw/my_dataset.h5ad
```

Start MLflow in a separate terminal before running experiments:

```bash
mlflow ui --backend-store-uri sqlite:///../mlflow_dbs/scRNAseq_lightgbm.db
```

The MLflow UI is available at:

```text
http://127.0.0.1:5000
```

### Run one dataset at one resolution

```bash
dvc exp run -f -n my_dataset_res_0p5 \
  -S scanpy.input_path=data/raw/my_dataset.h5ad \
  -S scanpy.dataset_name=my_dataset \
  -S scanpy.leiden_resolution=0.5
```

The `dataset_name` is used for MLflow run names, archived files, and figure folders.

### Run one dataset across multiple resolutions

To evaluate clustering behavior across resolutions, run the same dataset at several Leiden resolutions:

```bash
for res in 0.5 1.0 1.5 2.0 2.5
do
  name="my_dataset_res_${res}"

  dvc exp run -f -n $name \
    -S scanpy.input_path=data/raw/my_dataset.h5ad \
    -S scanpy.dataset_name=my_dataset \
    -S scanpy.leiden_resolution=$res
done
```

## Outputs

The latest DVC run writes to:

```text
data/processed/processed.h5ad
data/processed/annotated.h5ad
artifacts/models/lightgbm_model.txt
artifacts/metrics/metrics.json
artifacts/figures/
```

Archived outputs are saved by dataset and resolution:

```text
data/processed/archive/<dataset>_res_<resolution>.h5ad
artifacts/metrics/archive/<dataset>_res_<resolution>.json
results/figures/<dataset>/res_<resolution>/
```

The annotated `.h5ad` contains per-cell LightGBM outputs in `adata.obs`:

```text
predicted_cluster_lgbm
confidence_lgbm
entropy_lgbm
entropy_norm_lgbm
margin_lgbm
prediction_matches_leiden
low_confidence_lgbm
```

The full class probability matrix is stored in:

```text
adata.obsm["probs_lgbm"]
adata.uns["probs_lgbm_columns"]
```

## Interpreting metrics

LightGBM is trained to recover Leiden cluster labels from PCA features. The goal is not to replace Leiden clustering, but to assess how separable and stable the resulting clusters are.

High confidence, low entropy, and high margin suggest cells assigned to well-separated clusters.

Low confidence, high entropy, and low margin suggest ambiguous cells, possible cluster boundaries, transitional structure, or over-clustering.

When increasing Leiden resolution, useful warning signs include:

```text
decreasing macro F1
increasing entropy
decreasing margin
increasing low-confidence fraction
increasing mismatch between Leiden and LightGBM predictions
```

Additional development notes are available in `docs/dev_notes.md`.

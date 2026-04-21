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
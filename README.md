This project implements a reproducible pipeline for single-cell RNA-seq (scRNA-seq) analysis and classification.

In addition to standard clustering (Scanpy + Leiden), it introduces a supervised learning step (LightGBM) to assign cluster labels and provide **confidence scores for each cell**, offering a complementary perspective to traditional unsupervised analysis, for example by quantifying how the reliability of cluster assignments changes across different clustering resolutions.

The pipeline is designed for reproducibility and experimentation, using DVC for data and pipeline versioning, and MLflow for tracking model performance, confidence metrics, and experiment configurations across datasets and clustering parameters.
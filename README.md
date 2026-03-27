This project implements a reproducible pipeline for single-cell RNA-seq (scRNA-seq) analysis and classification.

In addition to standard clustering (Scanpy + Leiden), it introduces a supervised learning step (LightGBM) to assign cluster labels and provide **confidence scores for each cell**, offering a complementary perspective to traditional unsupervised analysis.

The pipeline is managed with DVC for reproducibility.

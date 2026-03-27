from __future__ import annotations

from pathlib import Path
from typing import Optional

import scanpy as sc


def run_scanpy_pipeline(
    output_path: str | Path,
    input_path: Optional[str | Path] = None,
    dataset_name: str = "pbmc3k",
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    leiden_resolution: float = 1.0,
) -> Path:
    """
    Run a standard Scanpy preprocessing workflow and save the processed AnnData.

    Parameters
    ----------
    output_path
        Path to the processed .h5ad file to save.
    input_path
        Optional path to an input .h5ad file. If None, use a built-in Scanpy dataset.
    dataset_name
        Built-in dataset name to load when input_path is None.
    n_top_genes
        Number of highly variable genes to keep.
    n_pcs
        Number of principal components to compute.
    n_neighbors
        Number of neighbors for graph construction.
    leiden_resolution
        Resolution parameter for Leiden clustering.

    Returns
    -------
    Path
        Path to the saved processed .h5ad file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if input_path is not None:
        adata = sc.read_h5ad(str(input_path))
    else:
        if dataset_name == "pbmc3k":
            adata = sc.datasets.pbmc3k()
        else:
            raise ValueError(f"Unsupported built-in dataset: {dataset_name}")

    print("Loaded AnnData:")
    print(adata)

    # Basic preprocessing
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    print("\nAfter HVG filtering:")
    print(adata)
    print(f"adata.X shape: {adata.X.shape}")

    # PCA
    sc.pp.pca(adata, n_comps=n_pcs)
    print(f"\nPCA shape: {adata.obsm['X_pca'].shape}")

    # Neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Clustering
    sc.tl.leiden(adata)

    # UMAP
    sc.tl.umap(adata)

    print("\nLeiden cluster sizes:")
    print(adata.obs["leiden"].value_counts())

    # Save processed AnnData
    adata.write(output_path)

    print(f"\nSaved processed AnnData to: {output_path}")
    return output_path
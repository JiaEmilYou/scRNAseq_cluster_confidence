from pathlib import Path
import scanpy as sc
import yaml

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    with open(project_root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    paths = params["paths"]

    input_path = project_root / paths["annotated"]
    output_dir = project_root / paths["figures"]
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_path)

    print("Loaded annotated AnnData:")
    print(adata)

    if "X_umap" not in adata.obsm:
        print("UMAP not found, computing...")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    sc.settings.figdir = str(output_dir)

    sc.pl.umap(adata, color="leiden", save="_leiden.png", show=False)
    sc.pl.umap(adata, color="predicted_cluster_lgbm", save="_lgbm_prediction.png", show=False)
    sc.pl.umap(adata, color="confidence_lgbm", cmap="viridis", save="_confidence.png", show=False)
    sc.pl.umap(adata, color="prediction_matches_leiden", save="_agreement.png", show=False)

    sc.pl.umap(
        adata,
        color=[
            "leiden",
            "predicted_cluster_lgbm",
            "confidence_lgbm",
            "prediction_matches_leiden"
        ],
        wspace=0.4,
        save="_combined.png",
        show=False
    )


if __name__ == "__main__":
    main()
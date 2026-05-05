from pathlib import Path
import scanpy as sc
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    with open(project_root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    paths = params["paths"]

    input_path = project_root / paths["annotated"]
    output_dir = project_root / paths["figures"]
    archive_base_dir = project_root / "results" / "figures"

    adata = sc.read_h5ad(input_path)

    pipeline_params = adata.uns.get("pipeline_params", {})
    dataset_name = pipeline_params.get("dataset_name", "unknown_dataset")
    leiden_resolution = pipeline_params.get("leiden_resolution", "unknown_resolution")

    resolution_str = str(leiden_resolution).replace(".", "p")
    output_dir = output_dir / dataset_name / f"res_{resolution_str}"
    archive_dir = archive_base_dir / dataset_name / f"res_{resolution_str}"

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

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

    required_cols = {
        "leiden",
        "predicted_cluster_lgbm",
        "confidence_lgbm",
        "entropy_lgbm",
        "margin_lgbm",
    }
    if required_cols.issubset(adata.obs.columns):
        obs = adata.obs.copy()

        if "entropy_norm_lgbm" not in obs:
            n_classes = obs["leiden"].astype(str).nunique()
            obs["entropy_norm_lgbm"] = obs["entropy_lgbm"] / np.log(n_classes)

        wrong_mask = (
            obs["predicted_cluster_lgbm"].astype(str)
            != obs["leiden"].astype(str)
        )
        wrong_fraction = float(wrong_mask.mean())
        tail_fraction = min(wrong_fraction * 2, 0.5)

        if wrong_mask.sum() == 0:
            tail_mask = np.zeros(adata.n_obs, dtype=bool)
        else:
            low_margin_cutoff = obs["margin_lgbm"].quantile(tail_fraction)
            high_entropy_cutoff = obs["entropy_norm_lgbm"].quantile(1 - tail_fraction)
            low_confidence_cutoff = obs["confidence_lgbm"].quantile(tail_fraction)

            tail_mask = (
                (obs["margin_lgbm"] <= low_margin_cutoff)
                | (obs["entropy_norm_lgbm"] >= high_entropy_cutoff)
                | (obs["confidence_lgbm"] <= low_confidence_cutoff)
            )

        highlight_mask = np.asarray(wrong_mask | tail_mask)
        coords = adata.obsm["X_umap"]
        leiden_codes = obs["leiden"].astype("category").cat.codes

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=leiden_codes,
            cmap="tab20",
            s=8,
            alpha=0.25,
            linewidths=0,
        )
        ax.scatter(
            coords[highlight_mask, 0],
            coords[highlight_mask, 1],
            c="crimson",
            marker="x",
            s=28,
            linewidths=0.9,
            label="wrong or uncertainty tail",
        )
        ax.set_title(
            "Wrong predictions or uncertainty-tail cells\n"
            f"wrong={wrong_fraction:.3f}, tail={tail_fraction:.3f}, "
            f"highlighted={highlight_mask.mean():.3f}"
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "umap_wrong_or_tail.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        missing = sorted(required_cols - set(adata.obs.columns))
        print(f"Skipping wrong/tail UMAP; missing columns: {missing}")

    for fig_file in output_dir.glob("*.png"):
        shutil.copy2(fig_file, archive_dir / fig_file.name)

    print(f"Saved figure copies to: {archive_dir}")


if __name__ == "__main__":
    main()

from pathlib import Path
import yaml
from scrnaseq_lightgbm.preprocessing.scanpy_pipeline import run_scanpy_pipeline


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # --- load params ---
    with open(project_root / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    scanpy_params = params["scanpy"]
    paths = params["paths"]

    # --- paths ---
    output_path = project_root / paths["processed"]

    # --- run pipeline ---
    input_path = scanpy_params["input_path"]
    if input_path is not None:
        input_path = project_root / input_path

    run_scanpy_pipeline(
        output_path=output_path,
        input_path=input_path,
        dataset_name=scanpy_params["dataset_name"],
        n_top_genes=scanpy_params["n_top_genes"],
        n_pcs=scanpy_params["n_pcs"],
        n_neighbors=scanpy_params["n_neighbors"],
        leiden_resolution=scanpy_params["leiden_resolution"],
    )


if __name__ == "__main__":
    main()
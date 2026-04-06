from pathlib import Path
import yaml
from scrnaseq_lightgbm.models.lightgbm_pipeline import run_lightgbm_pipeline


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # --- load params ---
    with open(project_root / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    paths = params["paths"]
    lgb_params = params["lightgbm"]
    

    input_path = project_root / paths["processed"]
    output_h5ad_path = project_root / paths["annotated"]
    model_path = project_root / paths["model"]
    metrics_path = project_root / paths["metrics"]

    eval_params = params["evaluation"]

    scanpy_params = params["scanpy"]

    metrics = run_lightgbm_pipeline(
        input_path=input_path,
        output_h5ad_path=output_h5ad_path,
        model_path=model_path,
        metrics_path=metrics_path,
        feature_key=lgb_params["feature_key"],
        label_key=lgb_params["label_key"],
        test_size=lgb_params["test_size"],
        random_state=lgb_params["random_state"],
        n_estimators=lgb_params["n_estimators"],
        learning_rate=lgb_params["learning_rate"],
        low_confidence_threshold=eval_params["low_confidence_threshold"],
        dataset_name=scanpy_params["dataset_name"],
        leiden_resolution=scanpy_params["leiden_resolution"],
    )

    print("\nFinal metrics summary:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
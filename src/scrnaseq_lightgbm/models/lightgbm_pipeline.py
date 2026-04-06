from pathlib import Path

import lightgbm as lgb
import numpy as np
import scanpy as sc
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("scrnaseq_lightgbm")

def run_lightgbm_pipeline(
    input_path: str | Path,
    output_h5ad_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    feature_key: str = "X_pca",
    label_key: str = "leiden",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    low_confidence_threshold: float = 0.6,
    dataset_name: str | None = None,
    leiden_resolution: float | None = None,
):
    input_path = Path(input_path)
    output_h5ad_path = Path(output_h5ad_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    output_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load data ---
    adata = sc.read_h5ad(str(input_path))

    # --- validate keys ---
    if feature_key not in adata.obsm:
        raise KeyError(f"Feature key '{feature_key}' not found in adata.obsm")
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' not found in adata.obs")
    
    # --- extract features and labels ---
    X = pd.DataFrame(
        adata.obsm[feature_key],
        index=adata.obs_names,
        columns=[f"{feature_key}_{i}" for i in range(adata.obsm[feature_key].shape[1])]
    )
    y = adata.obs[label_key].astype(str)

    print("Loaded processed AnnData:")
    print(adata)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print("\nLabel counts:")
    print(y.value_counts())


    #--- train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\nTrain/test split:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # ---train LightGBM model ---
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    n_clusters = int(adata.obs[label_key].nunique())
    run_name = f"{dataset_name}_res_{str(leiden_resolution).replace('.', 'p')}"
    with mlflow.start_run(run_name=run_name):

        # --- log parameters ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("low_confidence_threshold", low_confidence_threshold)
        mlflow.log_param("feature_key", feature_key)
        mlflow.log_param("label_key", label_key)    
        mlflow.log_metric("n_clusters", n_clusters)
        if dataset_name is not None:
            mlflow.log_param("dataset_name", dataset_name)

        if leiden_resolution is not None:
            mlflow.log_param("leiden_resolution", leiden_resolution)

        # --- train ---
        model.fit(X_train, y_train)

        # --- evaluate ---
        y_test_pred = model.predict(X_test)
        probs_test = model.predict_proba(X_test)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        macro_f1 = f1_score(y_test, y_test_pred, average="macro")
        mean_confidence_test = float(probs_test.max(axis=1).mean())
        low_confidence_fraction_test = float(
            (probs_test.max(axis=1) < low_confidence_threshold).mean()
        )

        # --- log metrics ---
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("mean_confidence_test", mean_confidence_test)
        mlflow.log_metric("low_confidence_fraction_test", low_confidence_fraction_test)


        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Mean test confidence: {mean_confidence_test:.4f}")
        print(f"Low-confidence fraction (test): {low_confidence_fraction_test:.4f}")

    

        # --- add predictions to adata ---
        probs_all = model.predict_proba(X)
        preds_all = model.predict(X)

        adata.obs["predicted_cluster_lgbm"] = preds_all.astype(str)
        adata.obs["confidence_lgbm"] = probs_all.max(axis=1)
        adata.obs["prediction_matches_leiden"] = (
            adata.obs[label_key].astype(str) == adata.obs["predicted_cluster_lgbm"]
        )
        adata.obs["low_confidence_lgbm"] = (
            adata.obs["confidence_lgbm"] < low_confidence_threshold
        )

        # --- save ouputs ---
        model.booster_.save_model(str(model_path))
        adata.write(str(output_h5ad_path))

        print(f"Saved model to: {model_path}")
        print(f"Saved annotated AnnData to: {output_h5ad_path}")

        # --- save metrics ---
        metrics = {
            "input_path": str(input_path),
            "output_h5ad_path": str(output_h5ad_path),
            "model_path": str(model_path),
            "dataset_name": dataset_name,
            "leiden_resolution": leiden_resolution,
            "feature_key": feature_key,
            "label_key": label_key,
            "n_cells": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_clusters": int(adata.obs[label_key].nunique()),
            "test_size": test_size,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "low_confidence_threshold": float(low_confidence_threshold),
            "test_accuracy": float(test_accuracy),
            "macro_f1": float(macro_f1),
            "mean_confidence_test": mean_confidence_test,
            "low_confidence_fraction_test": low_confidence_fraction_test,
            "mean_confidence_all_cells": float(np.mean(adata.obs["confidence_lgbm"])),
            "min_confidence_all_cells": float(np.min(adata.obs["confidence_lgbm"])),
            "max_confidence_all_cells": float(np.max(adata.obs["confidence_lgbm"])),
            "match_rate_all_cells": float(np.mean(adata.obs["prediction_matches_leiden"])),
            "low_confidence_fraction_all_cells": float(np.mean(adata.obs["low_confidence_lgbm"])),
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to: {metrics_path}")

        mlflow.sklearn.log_model(model, "lightgbm_model")
        mlflow.log_artifact(str(metrics_path))

    return metrics
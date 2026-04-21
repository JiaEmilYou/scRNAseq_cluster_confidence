from pathlib import Path


import lightgbm as lgb
import numpy as np
import scanpy as sc
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import mlflow
import mlflow.sklearn
import os
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
print(f"MLflow tracking URI: {tracking_uri}")
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
    cv_folds: int = 5,
    cv_repeats: int = 3,
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
    dataset_str = dataset_name if dataset_name is not None else "unknown_dataset"
    resolution_str = (
        str(leiden_resolution).replace(".", "p")
        if leiden_resolution is not None
        else "unknown_resolution"
    )
    run_name = f"{dataset_str}_res_{resolution_str}"
    with mlflow.start_run(run_name=run_name):

        # --- cross-validation evaluation ---
        min_class_size = y.value_counts().min()
        if cv_folds > min_class_size:
            raise ValueError(
                f"cv_folds={cv_folds} is too large for the smallest class "
                f"(size={min_class_size})."
            )
        
        cv_accuracies = []
        cv_macro_f1s = []
        cv_mean_confidences = []
        cv_low_conf_fractions = []
        cv_mean_entropy_norms = []
        cv_mean_margins = []


        for repeat_idx in range(cv_repeats):
            skf = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state + repeat_idx,
            )

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                X_train_cv = X.iloc[train_idx]
                X_test_cv = X.iloc[test_idx]
                y_train_cv = y.iloc[train_idx]
                y_test_cv = y.iloc[test_idx]

                model_cv = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=random_state + repeat_idx,
                )

                model_cv.fit(X_train_cv, y_train_cv)

                y_pred_cv = model_cv.predict(X_test_cv)
                probs_cv = model_cv.predict_proba(X_test_cv)

                acc_cv = accuracy_score(y_test_cv, y_pred_cv)
                f1_cv = f1_score(y_test_cv, y_pred_cv, average="macro")
                mean_conf_cv = float(probs_cv.max(axis=1).mean())
                low_conf_cv = float((probs_cv.max(axis=1) < low_confidence_threshold).mean())

                entropy_cv = -np.sum(probs_cv * np.log(probs_cv + 1e-12), axis=1)
                entropy_norm_cv = entropy_cv / np.log(probs_cv.shape[1])

                sorted_probs_cv = np.sort(probs_cv, axis=1)
                margin_cv = sorted_probs_cv[:, -1] - sorted_probs_cv[:, -2]

                cv_accuracies.append(acc_cv)
                cv_macro_f1s.append(f1_cv)
                cv_mean_confidences.append(mean_conf_cv)
                cv_low_conf_fractions.append(low_conf_cv)
                cv_mean_entropy_norms.append(float(np.mean(entropy_norm_cv)))
                cv_mean_margins.append(float(np.mean(margin_cv)))

        cv_accuracy_mean = float(np.mean(cv_accuracies))
        cv_accuracy_std = float(np.std(cv_accuracies))
        cv_macro_f1_mean = float(np.mean(cv_macro_f1s))
        cv_macro_f1_std = float(np.std(cv_macro_f1s))
        cv_mean_confidence_test_mean = float(np.mean(cv_mean_confidences))
        cv_mean_confidence_test_std = float(np.std(cv_mean_confidences))
        cv_low_confidence_fraction_test_mean = float(np.mean(cv_low_conf_fractions))
        cv_low_confidence_fraction_test_std = float(np.std(cv_low_conf_fractions))
        cv_mean_entropy_norm_mean = float(np.mean(cv_mean_entropy_norms))
        cv_mean_entropy_norm_std = float(np.std(cv_mean_entropy_norms))
        cv_mean_margin_mean = float(np.mean(cv_mean_margins))
        cv_mean_margin_std = float(np.std(cv_mean_margins))

        print(f"CV accuracy: {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")
        print(f"CV macro F1: {cv_macro_f1_mean:.4f} ± {cv_macro_f1_std:.4f}")
        print(
            f"CV mean confidence: "
            f"{cv_mean_confidence_test_mean:.4f} ± {cv_mean_confidence_test_std:.4f}"
        )
        print(
            f"CV low-confidence fraction: "
            f"{cv_low_confidence_fraction_test_mean:.4f} ± "
            f"{cv_low_confidence_fraction_test_std:.4f}"
        )
        print(
            f"CV mean normalized entropy: "
            f"{cv_mean_entropy_norm_mean:.4f} ± {cv_mean_entropy_norm_std:.4f}"
        )
        print(
            f"CV mean margin: "
            f"{cv_mean_margin_mean:.4f} ± {cv_mean_margin_std:.4f}"
        )

        # --- log parameters ---
        mlflow.log_param("cv_repeats", cv_repeats)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("low_confidence_threshold", low_confidence_threshold)
        mlflow.log_param("feature_key", feature_key)
        mlflow.log_param("label_key", label_key)  
        mlflow.log_param("cv_folds", cv_folds)

        mlflow.log_metric("cv_accuracy_mean", cv_accuracy_mean)
        mlflow.log_metric("cv_accuracy_std", cv_accuracy_std)
        mlflow.log_metric("cv_macro_f1_mean", cv_macro_f1_mean)
        mlflow.log_metric("cv_macro_f1_std", cv_macro_f1_std)
        mlflow.log_metric("cv_mean_confidence_test_mean", cv_mean_confidence_test_mean)
        mlflow.log_metric("cv_mean_confidence_test_std", cv_mean_confidence_test_std)
        mlflow.log_metric(
            "cv_low_confidence_fraction_test_mean",
            cv_low_confidence_fraction_test_mean,
        )
        mlflow.log_metric(
            "cv_low_confidence_fraction_test_std",
            cv_low_confidence_fraction_test_std,
        ) 
        mlflow.log_metric("cv_mean_entropy_norm_mean", cv_mean_entropy_norm_mean)
        mlflow.log_metric("cv_mean_entropy_norm_std", cv_mean_entropy_norm_std)
        mlflow.log_metric("cv_mean_margin_mean", cv_mean_margin_mean)
        mlflow.log_metric("cv_mean_margin_std", cv_mean_margin_std)         
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

        adata.obsm["probs_lgbm"] = probs_all
        adata.uns["probs_lgbm_columns"] = model.classes_.astype(str).tolist()

        adata.obs["predicted_cluster_lgbm"] = preds_all.astype(str)
        adata.obs["confidence_lgbm"] = probs_all.max(axis=1)

        entropy = -np.sum(probs_all * np.log(probs_all + 1e-12), axis=1)
        adata.obs["entropy_lgbm"] = entropy

        # --- margin (top1 - top2) ---
        sorted_probs = np.sort(probs_all, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        adata.obs["margin_lgbm"] = margin


        n_classes = probs_all.shape[1]
        entropy_norm = entropy / np.log(n_classes)

        mean_entropy_norm_all = float(np.mean(entropy_norm))
        p95_entropy_norm_all = float(np.percentile(entropy_norm, 95))
        p99_entropy_norm_all = float(np.percentile(entropy_norm, 99))

        adata.obs["entropy_norm_lgbm"] = entropy_norm



        mean_margin_all = float(np.mean(margin))
        p05_margin_all = float(np.percentile(margin, 5))



        adata.obs["prediction_matches_leiden"] = (
            adata.obs[label_key].astype(str) == adata.obs["predicted_cluster_lgbm"]
        )
        adata.obs["low_confidence_lgbm"] = (
            adata.obs["confidence_lgbm"] < low_confidence_threshold
        )

        mlflow.log_metric("mean_margin_all_cells", mean_margin_all)
        mlflow.log_metric("p05_margin_all_cells", p05_margin_all)
        mlflow.log_metric("mean_entropy_norm_all_cells", mean_entropy_norm_all)
        mlflow.log_metric("p95_entropy_norm_all_cells", p95_entropy_norm_all)
        mlflow.log_metric("p99_entropy_norm_all_cells", p99_entropy_norm_all)

        # --- save ouputs ---
        model.booster_.save_model(str(model_path))
        adata.write(str(output_h5ad_path))

        run_tag = run_name
        archive_path = Path("data/processed/archive") / f"{run_tag}.h5ad"
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        adata.write(str(archive_path))

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
            "cv_folds": cv_folds,
            "cv_accuracy_mean": cv_accuracy_mean,
            "cv_accuracy_std": cv_accuracy_std,
            "cv_macro_f1_mean": cv_macro_f1_mean,
            "cv_macro_f1_std": cv_macro_f1_std,
            "cv_mean_confidence_test_mean": cv_mean_confidence_test_mean,
            "cv_mean_confidence_test_std": cv_mean_confidence_test_std,
            "cv_low_confidence_fraction_test_mean": cv_low_confidence_fraction_test_mean,
            "cv_low_confidence_fraction_test_std": cv_low_confidence_fraction_test_std,
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
            "mean_margin_all_cells": mean_margin_all,
            "p05_margin_all_cells": p05_margin_all,
            "mean_entropy_norm_all_cells": mean_entropy_norm_all,
            "p95_entropy_norm_all_cells": p95_entropy_norm_all,
            "p99_entropy_norm_all_cells": p99_entropy_norm_all,
            "cv_repeats": cv_repeats,
            "cv_mean_entropy_norm_mean": cv_mean_entropy_norm_mean,
            "cv_mean_entropy_norm_std": cv_mean_entropy_norm_std,
            "cv_mean_margin_mean": cv_mean_margin_mean,
            "cv_mean_margin_std": cv_mean_margin_std,
            "archive_h5ad_path": str(archive_path),
        }

        metrics_archive_path = Path("artifacts/metrics/archive") / f"{run_tag}.json"
        metrics_archive_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_archive_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved archived metrics to: {metrics_archive_path}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to: {metrics_path}")

        mlflow.sklearn.log_model(model, "lightgbm_model")
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(output_h5ad_path))

    return metrics
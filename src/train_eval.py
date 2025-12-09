# src/train_eval.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils import compute_spearman


def compute_all_metrics(y_true, y_pred):
    """Compute all regression metrics used in CV."""
    spearman = compute_spearman(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Safe division for MAPE
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)

    return {
        "spearman": spearman,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def train_and_evaluate(X, y, models, n_splits=5):
    """
    Perform K-Fold CV on all models provided.
    Returns:
        results: dict of aggregated CV metrics
        trained_models: dict of final models trained on full dataset
    """

    results = {}
    trained_models = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")

        fold_metrics = {
            "spearman": [],
            "rmse": [],
            "mae": [],
            "r2": [],
            "mape": []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            # compute all metrics here
            metrics = compute_all_metrics(y_val, preds)

            fold_metrics["spearman"].append(metrics["spearman"])
            fold_metrics["rmse"].append(metrics["rmse"])
            fold_metrics["mae"].append(metrics["mae"])
            fold_metrics["r2"].append(metrics["r2"])
            fold_metrics["mape"].append(metrics["mape"])

            print(
                f"Fold {fold+1} - {model_name} spearman: {metrics['spearman']:.4f}"
            )

        # aggregate metrics
        results[model_name] = {
            "spearman_mean": np.mean(fold_metrics["spearman"]),
            "spearman_std": np.std(fold_metrics["spearman"]),

            "rmse_mean": np.mean(fold_metrics["rmse"]),
            "rmse_std": np.std(fold_metrics["rmse"]),

            "mae_mean": np.mean(fold_metrics["mae"]),
            "mae_std": np.std(fold_metrics["mae"]),

            "r2_mean": np.mean(fold_metrics["r2"]),
            "r2_std": np.std(fold_metrics["r2"]),

            "mape_mean": np.mean(fold_metrics["mape"]),
            "mape_std": np.std(fold_metrics["mape"]),
        }

        # Train final model on full dataset
        print(f"Training final model on full data: {model_name}")
        trained_models[model_name] = model.fit(X, y)

    return results, trained_models

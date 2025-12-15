# ================================
# bulk_test.py — FULL EVALUATION
# Compatible with unified pipeline (FeatureFusionBuilder object)
# ================================

import os
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from scipy.stats import spearmanr, pearsonr
import joblib

from src.preprocess import simple_clean
from src.features_fusion import FeatureFusionBuilder


# ----------------------------
# Category Mapping
# ----------------------------
def map_category(s):
    if s >= 0.85: return "Excellent"
    elif s >= 0.65: return "Strong"
    elif s >= 0.45: return "Moderate"
    elif s >= 0.25: return "Weak"
    else: return "NoMatch"


# ----------------------------
# MAIN BULK TEST
# ----------------------------
def run_bulk_test():

    INPUT_CSV = "C:/Users/INCHARA G/Title_Ranking_ML_model/datasets/real_world_dataset_2000_cleaned.csv"
    OUT_DIR = "outputs/bulk_test_results"
    PLOT_DIR = os.path.join(OUT_DIR, "plots")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"Loading dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"title", "abstract", "expected"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain: {required_cols}")

    df["title"] = df["title"].astype(str).apply(simple_clean)
    df["abstract"] = df["abstract"].astype(str).apply(simple_clean)

    print("Loading LightGBM model + artifacts...")
    model = joblib.load("outputs/models/lgbm.joblib")
    scaler = joblib.load("outputs/scaler.joblib")
    stats = json.load(open("outputs/target_stats.json"))
    t_mean, t_std = stats["mean"], stats["std"]

    print("Loading FeatureFusionBuilder object...")
    fb: FeatureFusionBuilder = joblib.load("outputs/feature_builder.joblib")
    fb.load_sbert()  # ensure SBERT is ready

    # -----------------------
    # Build feature matrix
    # -----------------------
    print("Building feature matrix...")
    X, _, _ = fb.build_feature_matrix(df, fit_tfidf=False, return_vectors=False)
    X = np.array(X)

    # Apply scaler
    X_scaled = scaler.transform(X)

    # -----------------------
    # Predictions
    # -----------------------
    print("Predicting...")
    pred_norm = model.predict(X_scaled)
    y_pred = (pred_norm * t_std) + t_mean

    df["predicted"] = y_pred
    df["pred_cat"] = df["predicted"].apply(map_category)

    y_true = df["expected"].astype(float).values
    true_cat = df["expected"].apply(map_category).values

    # -----------------------
    # Metrics
    # -----------------------
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    spearman_val = spearmanr(y_true, y_pred)[0]
    pearson_val = pearsonr(y_true, y_pred)[0]

    mask = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    numeric_accuracy = 100.0 - mape

    labels = ["Excellent", "Strong", "Moderate", "Weak", "NoMatch"]
    cm = confusion_matrix(true_cat, df["pred_cat"].values, labels=labels)

    category_accuracy = (true_cat == df["pred_cat"].values).mean() * 100.0

    # Save metrics JSON
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": spearman_val,
        "pearson": pearson_val,
        "mape": mape,
        "numeric_accuracy": numeric_accuracy,
        "category_accuracy": category_accuracy,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist()
    }

    json.dump(metrics, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=4)

    # -----------------------
    # Formatted Output
    # -----------------------
    print("\n===== NUMERIC REGRESSION METRICS =====")
    print(f"RMSE:     {rmse:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"R²:       {r2:.4f}")
    print(f"Spearman: {spearman_val:.4f}")
    print(f"Pearson:  {pearson_val:.4f}")
    print(f"MAPE:     {mape:.2f}%")
    print(f"Numeric Accuracy (1 - MAPE): {numeric_accuracy:.2f}%")

    print("\n===== CATEGORY ACCURACY =====")
    print(f"Category Accuracy: {category_accuracy:.2f}%\n")

    print("Confusion Matrix (rows=true, cols=pred) labels:", labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    # -----------------------
    # Plots
    # -----------------------
    errors = y_pred - y_true

    # Scatter
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.title("Predicted vs Expected")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "scatter.png"), dpi=150)
    plt.close()

    # Error distribution
    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=30)
    plt.title("Error Distribution")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "errors.png"), dpi=150)
    plt.close()

    # Residuals
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, errors, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predicted")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "residuals.png"), dpi=150)
    plt.close()

    df.to_csv(os.path.join(OUT_DIR, "predictions_bulk.csv"), index=False)

    print("\n✔ All outputs saved to:", OUT_DIR)
    print("✔ Plots saved to:", PLOT_DIR)
    print("Bulk test complete.")


# ----------------------------
# RUN SCRIPT
# ----------------------------
if __name__ == "__main__":
    run_bulk_test()

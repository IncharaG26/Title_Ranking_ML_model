"""
model_test_lgb.py — Test LightGBM model using predictions_lgbm.csv
Includes full evaluation: metrics + confusion matrix + graphs
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)

from src.preprocess import simple_clean
from src.features_fusion import FeatureFusionBuilder
import os


MODEL_PATH = "outputs/models/lgbm.joblib"
SCALER_PATH = "outputs/scaler.joblib"
TARGET_STATS_PATH = "outputs/target_stats.json"
PRED_CSV = "outputs/predictions_lgbm.csv"
FEATURE_BUILDER_PATH = "outputs/feature_builder.joblib"

PLOT_DIR = "outputs/model_test_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def map_category(s):
    if s >= 0.85: return "Excellent"
    if s >= 0.65: return "Strong"
    if s >= 0.45: return "Moderate"
    if s >= 0.25: return "Weak"
    return "NoMatch"


def main():

    print("Loading model and artifacts...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    stats = json.load(open(TARGET_STATS_PATH))
    t_mean, t_std = stats["mean"], stats["std"]

    print("Loading FeatureFusionBuilder object...")
    fb: FeatureFusionBuilder = joblib.load(FEATURE_BUILDER_PATH)
    fb.load_sbert()

    print("Loading predictions CSV...")
    df = pd.read_csv(PRED_CSV)

    df["title"] = df["title"].astype(str).apply(simple_clean)
    df["abstract"] = df["abstract"].astype(str).apply(simple_clean)

    print("Building feature matrix...")
    X, _, _ = fb.build_feature_matrix(df, fit_tfidf=False, return_vectors=False)
    X = np.array(X)

    print("Applying scaler...")
    X_scaled = scaler.transform(X)

    print("Running model predictions...")
    pred_norm = model.predict(X_scaled)
    y_pred = (pred_norm * t_std) + t_mean

    y_true = df["target_raw"].values

    # ======================
    # Metrics
    # ======================
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    spearman_val = spearmanr(y_true, y_pred)[0]
    pearson_val  = pearsonr(y_true, y_pred)[0]

    print("\n===== MODEL TEST METRICS =====")
    print(f"RMSE:     {rmse:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"R²:       {r2:.4f}")
    print(f"Spearman: {spearman_val:.4f}")
    print(f"Pearson:  {pearson_val:.4f}")

    # ======================
    # Category evaluation
    # ======================
    pred_cat = [map_category(s) for s in y_pred]
    true_cat = [map_category(s) for s in y_true]

    acc = (np.array(pred_cat) == np.array(true_cat)).mean() * 100

    labels = ["Excellent", "Strong", "Moderate", "Weak", "NoMatch"]
    cm = confusion_matrix(true_cat, pred_cat, labels=labels)

    print("\n===== CATEGORY ACCURACY =====")
    print(f"Accuracy: {acc:.2f}%")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())


    # ========================================
    # PLOTS
    # ========================================
    # Scatter plot
    plt.figure(figsize=(7,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn, mx = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs True")
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/scatter_pred_true.png", dpi=150)
    plt.close()

    # Residual plot
    errors = y_pred - y_true
    plt.figure(figsize=(7,6))
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Error)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/residuals.png", dpi=150)
    plt.close()

    # Error distribution
    plt.figure(figsize=(7,5))
    plt.hist(errors, bins=30, alpha=0.8)
    plt.title("Error Distribution (Pred - True)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/error_distribution.png", dpi=150)
    plt.close()

    # Prediction distribution
    plt.figure(figsize=(7,5))
    plt.hist(y_pred, bins=30, alpha=0.8)
    plt.title("Prediction Value Distribution")
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/pred_distribution.png", dpi=150)
    plt.close()

    print("\n✔ Saved all plots to:", PLOT_DIR)


if __name__ == "__main__":
    main()

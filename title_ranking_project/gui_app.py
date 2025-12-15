# gui_app.py — Updated Tkinter GUI (Unified Pipeline Compatible)

import tkinter as tk
from tkinter import ttk, scrolledtext
import joblib
import pandas as pd
import numpy as np
import json
import os

from src.preprocess import simple_clean
from src.features_fusion import FeatureFusionBuilder


# ======================================================
# PATHS
# ======================================================
MODEL_PATH = "outputs/models/lgbm.joblib"
SCALER_PATH = "outputs/scaler.joblib"
TARGET_STATS_PATH = "outputs/target_stats.json"
FEATURE_BUILDER_PATH = "outputs/feature_builder.joblib"


# ======================================================
# LOAD ARTIFACTS
# ======================================================
print("Loading LightGBM model...")
model = joblib.load(MODEL_PATH)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Loading target statistics...")
stats = json.load(open(TARGET_STATS_PATH))
t_mean, t_std = stats["mean"], stats["std"]

print("Loading FeatureFusionBuilder object...")
fb: FeatureFusionBuilder = joblib.load(FEATURE_BUILDER_PATH)
fb.load_sbert()   # important! loads SBERT model


# ======================================================
# CATEGORY MAPPING
# ======================================================
def map_category(score):
    if score >= 0.85: return "Excellent Match"
    elif score >= 0.65: return "Strong Match"
    elif score >= 0.45: return "Moderate Match"
    elif score >= 0.25: return "Weak Match"
    return "No Match"


# ======================================================
# PREDICT FUNCTION
# ======================================================
def predict_score():

    title = title_input.get("1.0", tk.END).strip()
    abstract = abstract_input.get("1.0", tk.END).strip()

    if not title or not abstract:
        result_label.config(text="❌ Missing title or abstract")
        return

    # Clean input
    df = pd.DataFrame([{
        "title": simple_clean(title),
        "abstract": simple_clean(abstract)
    }])

    # Build fused features (NO TF-IDF FITTING)
    X, _, _ = fb.build_feature_matrix(df, fit_tfidf=False, return_vectors=False)
    X = np.array(X)

    # Apply scaler
    X_scaled = scaler.transform(X)

    # Predict normalized → denormalize
    pred_norm = model.predict(X_scaled)[0]
    pred = float(pred_norm * t_std + t_mean)

    # Category
    cat = map_category(pred)

    # Display
    result_label.config(
        text=(
            f"Predicted Score: {pred:.4f}\n"
            f"Category: {cat}"
        )
    )


# ======================================================
# TKINTER UI
# ======================================================
root = tk.Tk()
root.title("AI Title–Abstract Relevance Predictor (Unified LightGBM Pipeline)")
root.geometry("820x620")

ttk.Label(root, text="Title:", font=("Arial", 12)).pack()
title_input = scrolledtext.ScrolledText(root, height=4, width=100)
title_input.pack()

ttk.Label(root, text="Abstract:", font=("Arial", 12)).pack()
abstract_input = scrolledtext.ScrolledText(root, height=12, width=100)
abstract_input.pack()

ttk.Button(root, text="Predict Score", command=predict_score).pack(pady=12)

result_label = ttk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()

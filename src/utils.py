# src/utils.py
import os
import pandas as pd
import joblib
from scipy.stats import spearmanr
import numpy as np

def save_predictions_df(df, path="outputs/predictions_all.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved:", path)

def compute_spearman(y_true, y_pred):
    r = spearmanr(y_true, y_pred).correlation
    if np.isnan(r):
        return 0.0
    return float(r)

def save_joblib(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

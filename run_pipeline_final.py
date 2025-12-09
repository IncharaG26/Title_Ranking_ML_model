"""
Unified training pipeline using ArXiv Kaggle metadata.

- Uses only the ArXiv dataset file:
    arxiv-metadata-oai-snapshot.json

- Reads only the first 20,000 records (rows) from the JSON file
  to keep the pipeline memory-safe.

- Steps:
    1. Load first 20,000 ArXiv records (title + abstract)
    2. Clean text and remove duplicates
    3. Build SBERT-based target score (cosine similarity)
    4. Build fused feature matrix (TF-IDF + SVD + SBERT) via FeatureFusionBuilder
    5. Train LightGBM with K-Fold cross validation
    6. Train final model on all data and save:
         outputs/target_stats.json
         outputs/scaler.joblib
         outputs/models/lgbm.joblib
         outputs/feature_builder.joblib
         outputs/predictions_lgbm.csv
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMRegressor

from src.preprocess import simple_clean
from src.features_fusion import FeatureFusionBuilder

# ============================================================
# CONFIGURATION
# ============================================================

# Absolute path to your ArXiv JSON file (Kaggle dump)
ARXIV_JSON_PATH = r"C:\Users\INCHARA G\OneDrive\Desktop\aiml_arxiv\datasets\aixrv_dataset\arxiv-metadata-oai-snapshot.json"

OUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUT_DIR, "models")
FEATURE_BUILDER_PATH = os.path.join(OUT_DIR, "feature_builder.joblib")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# SBERT model and training settings
SBERT_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 64

# Only use first 20,000 rows from ArXiv dataset
MAX_ARXIV_ROWS = 20000

RANDOM_STATE = SEED
np.random.seed(RANDOM_STATE)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def stream_jsonl_to_df(path, max_rows=None):
    """
    Stream a large JSON / JSONL file line-by-line and build a DataFrame
    for at most `max_rows` lines.
    This avoids loading the entire file into memory.
    """
    rows = []
    print(f"Streaming JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError:
                # skip malformed lines
                continue

    print(f"  -> Collected {len(rows)} rows from {os.path.basename(path)}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def ensure_columns(
    df,
    title_cols=("title", "paper_title"),
    abstract_cols=("abstract", "paper_abstract"),
):
    """
    Normalize column names so that we always end up with 'title' and 'abstract'.
    If present, only those two columns are kept, with NA rows dropped.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["title", "abstract"])

    for alt in title_cols:
        if alt in df.columns and "title" not in df.columns:
            df = df.rename(columns={alt: "title"})
    for alt in abstract_cols:
        if alt in df.columns and "abstract" not in df.columns:
            df = df.rename(columns={alt: "abstract"})

    if "title" in df.columns and "abstract" in df.columns:
        df = df[["title", "abstract"]].dropna().reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["title", "abstract"])

    return df


def batch_encode(model, texts, batch_size=BATCH_SIZE):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch = texts[i : i + batch_size]
        embs.append(model.encode(batch, convert_to_numpy=True))
    if len(embs) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    return np.vstack(embs)


# ============================================================
# 1. LOAD FIRST 20,000 ROWS FROM ARXIV
# ============================================================

print("\n=== STEP 1: LOAD FIRST 20,000 ROWS FROM ARXIV DATASET ===")
if not os.path.exists(ARXIV_JSON_PATH):
    raise FileNotFoundError(f"ArXiv JSON file not found at: {ARXIV_JSON_PATH}")

df_all = stream_jsonl_to_df(ARXIV_JSON_PATH, max_rows=MAX_ARXIV_ROWS)
df_all = ensure_columns(df_all)
print("Total ArXiv rows loaded (before cleaning):", len(df_all))


# ============================================================
# 2. CLEANING & DEDUPLICATION
# ============================================================

print("\n=== STEP 2: CLEANING & DEDUPLICATION ===")

df_all["title"] = df_all["title"].astype(str)
df_all["abstract"] = df_all["abstract"].astype(str)

# Remove empty titles/abstracts
df_all = df_all[
    (df_all["title"].str.strip() != "") &
    (df_all["abstract"].str.strip() != "")
]
initial_count = len(df_all)

# Remove duplicates by exact title+abstract signature
df_all["__sig"] = (
    df_all["title"].str.strip().str.lower()
    + " ||| "
    + df_all["abstract"].str.strip().str.lower()
)
df_all = df_all.drop_duplicates(subset="__sig").drop(columns="__sig").reset_index(drop=True)

print(f"Rows after cleaning & deduplication: {len(df_all)} (was {initial_count})")

# Shuffle
df_all = df_all.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
df_merged = df_all
print("Final training candidate size:", len(df_merged))


# ============================================================
# 3. TEXT CLEANING
# ============================================================

print("\n=== STEP 3: TEXT CLEANING ===")
df_merged["title"] = df_merged["title"].astype(str).apply(simple_clean)
df_merged["abstract"] = df_merged["abstract"].astype(str).apply(simple_clean)


# ============================================================
# 4. SBERT ENCODING & TARGET SCORE
# ============================================================

print("\n=== STEP 4: SBERT ENCODING & TARGET BUILDING ===")
sbert = SentenceTransformer(SBERT_MODEL)

titles = df_merged["title"].tolist()
abstracts = df_merged["abstract"].tolist()

title_emb = batch_encode(sbert, titles, batch_size=BATCH_SIZE)
abs_emb = batch_encode(sbert, abstracts, batch_size=BATCH_SIZE)

# Safe cosine similarity
num = np.sum(title_emb * abs_emb, axis=1)
den = np.linalg.norm(title_emb, axis=1) * np.linalg.norm(abs_emb, axis=1)
den = np.where(den == 0, 1e-9, den)
sbert_cos = num / den

# Map cosine [0,1] to target score [0.10, 0.95]
target_raw = 0.10 + np.clip(sbert_cos, 0, 1) * 0.85
df_merged["target_raw"] = target_raw
print(
    "Target (raw) stats: mean =",
    float(df_merged["target_raw"].mean()),
    " std =",
    float(df_merged["target_raw"].std()),
)


# ============================================================
# 5. FEATURE FUSION (TF-IDF, SVD, SBERT)
# ============================================================

print("\n=== STEP 5: BUILD FEATURE MATRIX WITH FeatureFusionBuilder ===")
fb = FeatureFusionBuilder(
    use_sbert=True,
    sbert_model_name=SBERT_MODEL,
    batch_size=BATCH_SIZE,
)

# Fit TF-IDF/SVD on training set
X, feat_names, _ = fb.build_feature_matrix(df_merged, fit_tfidf=True, return_vectors=False)
X = np.array(X)
print("Feature matrix shape:", X.shape)

# Save feature builder
fb.save(OUT_DIR)
joblib.dump(fb, FEATURE_BUILDER_PATH)
print(f"Saved feature builder to {FEATURE_BUILDER_PATH}")


# ============================================================
# 6. TARGET NORMALIZATION & SCALING
# ============================================================

print("\n=== STEP 6: TARGET NORMALIZATION & SCALING ===")
t_mean = float(df_merged["target_raw"].mean())
t_std = float(df_merged["target_raw"].std(ddof=0) if df_merged["target_raw"].std(ddof=0) > 0 else 1.0)
y_norm = (df_merged["target_raw"].values - t_mean) / t_std

with open(os.path.join(OUT_DIR, "target_stats.json"), "w") as f:
    json.dump({"mean": t_mean, "std": t_std}, f)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
print("Saved scaler to outputs/scaler.joblib")


# ============================================================
# 7. TRAIN LIGHTGBM WITH K-FOLD
# ============================================================

print("\n=== STEP 7: TRAIN LIGHTGBM (K-FOLD) ===")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rmse_list, spearman_list = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    print(f"\n-- Fold {fold}")
    model = LGBMRegressor(
        learning_rate=0.05,
        n_estimators=1500,
        num_leaves=32,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled[tr_idx], y_norm[tr_idx])

    pred_val = model.predict(X_scaled[val_idx]) * t_std + t_mean
    rmse = np.sqrt(mean_squared_error(df_merged["target_raw"].values[val_idx], pred_val))
    sp = float(spearmanr(df_merged["target_raw"].values[val_idx], pred_val)[0])

    print(f"Fold {fold} RMSE: {rmse:.4f}  Spearman: {sp:.4f}")
    rmse_list.append(rmse)
    spearman_list.append(sp)

print("\nLightGBM CV mean RMSE:", np.mean(rmse_list))
print("LightGBM CV mean Spearman:", np.mean(spearman_list))


# ============================================================
# 8. TRAIN FINAL MODEL & SAVE
# ============================================================

print("\n=== STEP 8: TRAIN FINAL MODEL & SAVE ARTIFACTS ===")
final_model = LGBMRegressor(
    learning_rate=0.03,
    n_estimators=2000,
    num_leaves=32,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
final_model.fit(X_scaled, y_norm)

joblib.dump(final_model, os.path.join(MODEL_DIR, "lgbm.joblib"))
print("Saved LightGBM model to:", os.path.join(MODEL_DIR, "lgbm.joblib"))


# ============================================================
# 9. SAVE PREDICTIONS & META
# ============================================================

print("\n=== STEP 9: SAVE PREDICTIONS & METADATA ===")
df_merged["pred_lgbm"] = final_model.predict(X_scaled) * t_std + t_mean
pred_out = os.path.join(OUT_DIR, "predictions_lgbm.csv")
df_merged.to_csv(pred_out, index=False)
print("Saved predictions to:", pred_out)

meta = {
    "rows_used": len(df_merged),
    "feature_count": X.shape[1],
    "sbert_model": SBERT_MODEL,
    "target_mean": t_mean,
    "target_std": t_std,
}
with open(os.path.join(OUT_DIR, "pipeline_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n==============================")
print("ArXiv-only pipeline complete — ALL ARTIFACTS SAVED")
print("You can now run (if present in repo):")
print("   python model_test_lgb.py")
print("   python bulk_test.py")
print("==============================")

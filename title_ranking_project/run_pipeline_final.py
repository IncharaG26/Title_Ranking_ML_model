# run_pipeline_final.py  (or run_pipeline_unified.py)
"""
Unified training pipeline (ArXiv-only, memory-safe version)

- Loads ONLY ArXiv metadata downloaded from Kaggle, stored under:
    datasets/arxiv_dataset/
  or (fallback if you named it "aixrv_dataset"):
    datasets/aixrv_dataset/

- Streams the large ArXiv JSON file line-by-line to avoid MemoryError
  and only loads up to MAX_ARXIV_ROWS rows into memory.

- Ensures NO overlap with evaluation set (if present):
    datasets/real_world_dataset_2000_cleaned.csv

- Cleans text, deduplicates, builds SBERT + fusion features (FeatureFusionBuilder)
- Trains LightGBM with K-Fold and saves artifacts:
    outputs/target_stats.json
    outputs/scaler.joblib
    outputs/models/lgbm.joblib
    outputs/feature_builder.joblib
    outputs/predictions_lgbm.csv

- Usage: python run_pipeline_final.py
"""

import os
import json
import time
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

# -------------------------
# CONFIG
# -------------------------
# Resolve `datasets` directory relative to the script location (repo root is parent of this folder).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATASET_FOLDER = os.path.join(REPO_ROOT, "datasets")

# Fallback to './datasets' from current working directory if the above doesn't exist
if not os.path.exists(DATASET_FOLDER):
    alt = os.path.join(os.getcwd(), "datasets")
    if os.path.exists(alt):
        DATASET_FOLDER = alt
    else:
        print(
            f"Warning: resolved datasets folder '{DATASET_FOLDER}' does not exist."
            f" Attempting to use './datasets' from CWD as fallback."
        )
        DATASET_FOLDER = os.path.join(os.getcwd(), "datasets")

EVAL_TEST_PATH = os.path.join(DATASET_FOLDER, "real_world_dataset_2000_cleaned.csv")
OUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUT_DIR, "models")
FEATURE_BUILDER_PATH = os.path.join(OUT_DIR, "feature_builder.joblib")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SBERT_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
SEED = 42
N_SPLITS = 5
SBATCH = 64

# how many rows we will read from BIG ArXiv JSON (to avoid MemoryError)
MAX_ARXIV_ROWS = 20000

RANDOM_STATE = SEED
np.random.seed(RANDOM_STATE)

# -------------------------
# Utility helpers
# -------------------------
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
                # skip malformed line
                continue

    print(f"  → Collected {len(rows)} rows from {os.path.basename(path)}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_arxiv_only(dataset_root, max_rows=MAX_ARXIV_ROWS):
    """
    Load ONLY the ArXiv dataset downloaded from Kaggle.

    Expected directory structure:
        datasets/arxiv_dataset/
    or (fallback)
        datasets/aixrv_dataset/

    It will:
      - stream large .json/.jsonl files line-by-line up to `max_rows`
      - read any .csv files directly (assuming they are smaller)
    """
    # Prefer correct spelling "arxiv_dataset"
    arxiv_dir = os.path.join(dataset_root, "arxiv_dataset")
    if not os.path.exists(arxiv_dir):
        # Fallback to the folder name you mentioned: "aixrv_dataset"
        arxiv_dir = os.path.join(dataset_root, "aixrv_dataset")

    if not os.path.exists(arxiv_dir):
        print(
            "ERROR: ArXiv dataset folder not found.\n"
            "Expected 'datasets/arxiv_dataset' or 'datasets/aixrv_dataset'."
        )
        return []

    dfs = []
    print("Loading ONLY ArXiv dataset from:", arxiv_dir)

    for fn in os.listdir(arxiv_dir):
        p = os.path.join(arxiv_dir, fn)
        fname_lower = fn.lower()

        # For the big Kaggle file like 'arxiv-metadata-oai-snapshot.json'
        if fname_lower.endswith(".json") or fname_lower.endswith(".jsonl"):
            print("→ Streaming ArXiv JSON:", p)
            df_part = stream_jsonl_to_df(p, max_rows=max_rows)
            dfs.append(df_part)

        elif fname_lower.endswith(".csv"):
            print("→ Loading ArXiv CSV:", p)
            # For CSV we assume it's smaller; if it's huge you can also sample with nrows=max_rows
            df_part = pd.read_csv(p)
            if max_rows is not None and len(df_part) > max_rows:
                df_part = df_part.iloc[:max_rows].reset_index(drop=True)
                print(f"  → Trimmed CSV to first {max_rows} rows.")
            dfs.append(df_part)

        else:
            # ignore other file types
            continue

    print("Total ArXiv file parts loaded:", len(dfs))
    return dfs


def remove_test_overlap(df_train, eval_path):
    """Remove rows whose title/abstract overlap with an external evaluation CSV, if provided."""
    if not os.path.exists(eval_path):
        print("Evaluation file not found, skipping overlap check:", eval_path)
        return df_train
    eval_df = pd.read_csv(eval_path)
    if "title" not in eval_df.columns or "abstract" not in eval_df.columns:
        print("Evaluation file missing 'title'/'abstract' columns, skipping overlap removal.")
        return df_train

    eval_titles = set(eval_df["title"].astype(str).str.strip().str.lower())
    eval_abstracts = set(eval_df["abstract"].astype(str).str.strip().str.lower())
    initial = len(df_train)
    mask_title = df_train["title"].astype(str).str.strip().str.lower().isin(eval_titles)
    mask_abs = df_train["abstract"].astype(str).str.strip().str.lower().isin(eval_abstracts)
    df_train_clean = df_train[~(mask_title | mask_abs)].reset_index(drop=True)
    removed = initial - len(df_train_clean)
    print(f"Removed {removed} rows that overlapped with evaluation set.")
    return df_train_clean


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
    # keep only title & abstract and drop rows missing them
    if "title" in df.columns and "abstract" in df.columns:
        df = df[["title", "abstract"]].dropna().reset_index(drop=True)
    else:
        # create empty frame with two cols if not present, to avoid errors later
        df = pd.DataFrame(columns=["title", "abstract"])
    return df


def batch_encode(model, texts, batch_size=SBATCH):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch = texts[i : i + batch_size]
        embs.append(model.encode(batch, convert_to_numpy=True))
    return np.vstack(embs) if len(embs) else np.zeros((0, model.get_sentence_embedding_dimension()))


# -------------------------
# 1. Load & merge datasets (ArXiv only)
# -------------------------
print("\n=== STEP 1: LOAD & MERGE DATASETS (ARXIV ONLY) ===")
datasets = load_arxiv_only(DATASET_FOLDER, max_rows=MAX_ARXIV_ROWS)
if datasets:
    df_all = pd.concat(datasets, ignore_index=True, sort=False)
else:
    df_all = pd.DataFrame(columns=["title", "abstract"])

# Normalize column names & pick title/abstract
df_all = ensure_columns(df_all)

# For this assignment we use ONLY ArXiv, so merged = df_all
df_merged = df_all.copy()
print("Total ArXiv rows (before cleaning):", len(df_merged))

# -------------------------
# 2. Clean, drop NA, dedupe
# -------------------------
print("\n=== STEP 2: CLEANING & DEDUPLICATION ===")

# Ensure correct columns again (safety)
df_merged = ensure_columns(df_merged)

# Drop rows with empty title/abstract
df_merged["title"] = df_merged["title"].astype(str)
df_merged["abstract"] = df_merged["abstract"].astype(str)
df_merged = df_merged[
    (df_merged["title"].str.strip() != "") & (df_merged["abstract"].str.strip() != "")
]
initial_count = len(df_merged)

# Remove duplicates by exact title+abstract
df_merged["__sig"] = (
    df_merged["title"].str.strip().str.lower()
    + " ||| "
    + df_merged["abstract"].str.strip().str.lower()
)
df_merged = (
    df_merged.drop_duplicates(subset="__sig")
    .drop(columns="__sig")
    .reset_index(drop=True)
)

# Remove any samples that overlap with evaluation test set (if exists)
df_merged = remove_test_overlap(df_merged, EVAL_TEST_PATH)

print(f"Rows after cleaning/dedup/overlap removal: {len(df_merged)} (was {initial_count})")

# Optionally shuffle
df_merged = df_merged.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

# -------------------------
# 3. Take subset for training if extremely large (safety)
# -------------------------
TARGET_MIN_ROWS = 10000
if len(df_merged) < TARGET_MIN_ROWS:
    print(
        f"Warning: merged dataset has only {len(df_merged)} rows (< {TARGET_MIN_ROWS}). "
        "Proceeding with available data."
    )
else:
    # If merged is larger, keep up to 15000 to limit memory, prefer random subset (already shuffled)
    df_merged = df_merged.iloc[:15000].reset_index(drop=True)
    print("Trimmed merged dataset to top 15000 for processing.")

print("Final training candidate size:", len(df_merged))

# -------------------------
# 4. Clean text fields using your preprocess.simple_clean
# -------------------------
print("\n=== STEP 3: TEXT CLEANING ===")
df_merged["title"] = df_merged["title"].astype(str).apply(simple_clean)
df_merged["abstract"] = df_merged["abstract"].astype(str).apply(simple_clean)

# -------------------------
# 5. SBERT encoding + target build
# -------------------------
print("\n=== STEP 4: SBERT ENCODING & TARGET BUILDING ===")
sbert = SentenceTransformer(SBERT_MODEL)

titles = df_merged["title"].tolist()
abstracts = df_merged["abstract"].tolist()

title_emb = batch_encode(sbert, titles, batch_size=SBATCH)
abs_emb = batch_encode(sbert, abstracts, batch_size=SBATCH)

# Safe cosine similarity
num = np.sum(title_emb * abs_emb, axis=1)
den = np.linalg.norm(title_emb, axis=1) * np.linalg.norm(abs_emb, axis=1)
den = np.where(den == 0, 1e-9, den)
sbert_cos = num / den

# Map to your expected scale [0.10, 0.95]
target_raw = 0.10 + np.clip(sbert_cos, 0, 1) * 0.85
df_merged["target_raw"] = target_raw
print(
    "Target (raw) stats: mean=",
    float(df_merged["target_raw"].mean()),
    "std=",
    float(df_merged["target_raw"].std()),
)

# -------------------------
# 6. Feature fusion (TF-IDF, SVD, SBERT features, etc.)
# -------------------------
print("\n=== STEP 5: BUILD FEATURE MATRIX WITH FeatureFusionBuilder ===")
fb = FeatureFusionBuilder(
    use_sbert=True,
    sbert_model_name=SBERT_MODEL,
    batch_size=SBATCH,
)

# Fit TF-IDF/SVD on training merged set (fit_tfidf=True)
X, feat_names, _ = fb.build_feature_matrix(df_merged, fit_tfidf=True, return_vectors=False)
X = np.array(X)
print("Feature matrix shape:", X.shape)

# Save feature builder components for reuse
fb.save(OUT_DIR)
joblib.dump(fb, FEATURE_BUILDER_PATH)  # convenience; fb.save already saves components in OUT_DIR

# -------------------------
# 7. Target normalization & scaler
# -------------------------
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

# -------------------------
# 8. Train LightGBM with K-Fold
# -------------------------
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

# -------------------------
# 9. Train final model on all data & save
# -------------------------
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

# -------------------------
# 10. Save predictions & metadata
# -------------------------
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
print("Unified ArXiv-only pipeline complete — ALL ARTIFACTS SAVED")
print("You can now run (if present in repo):")
print("   python model_test_lgb.py")
print("   python bulk_test.py")
print("==============================")

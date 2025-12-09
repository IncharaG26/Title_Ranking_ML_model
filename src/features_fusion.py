# src/features_fusion.py
import joblib
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

from src.preprocess import simple_clean


class FeatureFusionBuilder:
    """
    Unified and stable feature builder used for:
      - run_pipeline_final.py (LightGBM training)
      - model_test_lgb.py      (evaluation)
      - gui_app.py             (real-time prediction)

    Produces EXACT 10-feature vector used for LightGBM training.
    """

    def __init__(self, use_sbert=True, sbert_model_name=None, batch_size=32):
        self.use_sbert = use_sbert
        self.sbert_model_name = sbert_model_name
        self.batch_size = batch_size

        # Model components
        self.sbert = None
        self.tfidf = None
        self.svd = None
        self.scaler = None

    # ---------------------------------------------------------
    # SBERT Loader
    # ---------------------------------------------------------
    def load_sbert(self):
        if self.use_sbert and self.sbert is None:
            print("Loading SBERT:", self.sbert_model_name)
            self.sbert = SentenceTransformer(self.sbert_model_name)

    # ---------------------------------------------------------
    # SBERT encoder for titles/abstracts
    # ---------------------------------------------------------
    def encode_sbert(self, texts):
        if self.sbert is None:
            self.load_sbert()
        return np.array(
            self.sbert.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )
        )

    # ---------------------------------------------------------
    # SAVE components
    # ---------------------------------------------------------
    def save_components(self, out_dir):
        obj = {
            "tfidf": self.tfidf,
            "svd": self.svd,
            "scaler": self.scaler
        }
        out_path = os.path.join(out_dir, "feature_builder.joblib")
        joblib.dump(obj, out_path)
        print("✔ Saved feature components to:", out_path)

    # ---------------------------------------------------------
    # NEW: save wrapper — required by run_pipeline_final.py
    # ---------------------------------------------------------
    def save(self, out_dir):
        """Compatibility wrapper so scripts calling fb.save() still work."""
        self.save_components(out_dir)

    # ---------------------------------------------------------
    # LOAD components (TF-IDF, SVD & Scaler)
    # ---------------------------------------------------------
    def load_components(self, directory):
        path = os.path.join(directory, "feature_builder.joblib")
        data = joblib.load(path)  # this is a DICT

        self.tfidf = data["tfidf"]
        self.svd = data["svd"]
        self.scaler = data["scaler"]

        print("✔ Loaded TF-IDF, SVD & Scaler from:", path)

    # ---------------------------------------------------------
    # Build the EXACT 10-feature vector used in LightGBM
    # ---------------------------------------------------------
    def build_feature_matrix(self, df, fit_tfidf=False, return_vectors=False):

        titles = df["title"].astype(str).apply(simple_clean).tolist()
        abstracts = df["abstract"].astype(str).apply(simple_clean).tolist()

        # -----------------------------------------------------
        # (1) SBERT embeddings
        # -----------------------------------------------------
        self.load_sbert()

        sbert_title = self.encode_sbert(titles)
        sbert_abs = self.encode_sbert(abstracts)

        # cosine similarity
        sbert_cos = np.sum(sbert_title * sbert_abs, axis=1) / (
            np.linalg.norm(sbert_title, axis=1) * np.linalg.norm(sbert_abs, axis=1)
        )

        # -----------------------------------------------------
        # (2) TF-IDF → SVD
        # -----------------------------------------------------
        combined_texts = [t + " " + a for t, a in zip(titles, abstracts)]

        if fit_tfidf:
            self.tfidf = TfidfVectorizer(max_features=20000)
            tfidf_mat = self.tfidf.fit_transform(combined_texts)

            self.svd = TruncatedSVD(n_components=300)
            svd_feat = self.svd.fit_transform(tfidf_mat)
        else:
            tfidf_mat = self.tfidf.transform(combined_texts)
            svd_feat = self.svd.transform(tfidf_mat)

        svd_first3 = svd_feat[:, :3]

        # -----------------------------------------------------
        # (3) Hand-crafted features
        # -----------------------------------------------------
        len_title = np.array([len(t.split()) for t in titles])
        len_abs = np.array([len(a.split()) for a in abstracts])
        len_ratio = len_title / (len_abs + 1)

        overlap = np.array([
            len(set(t.split()).intersection(set(a.split())))
            for t, a in zip(titles, abstracts)
        ])

        overlap_ratio = overlap / (len_abs + 1)

        # -----------------------------------------------------
        # (4) Final 10-feature matrix
        # -----------------------------------------------------
        X = np.column_stack([
            sbert_cos,       # 1
            svd_first3,      # 3 → total 4
            len_title,       # 5
            len_abs,         # 6
            len_ratio,       # 7
            overlap,         # 8
            overlap_ratio    # 9
        ])

        # Note: 9 features — your SVD gives 3 comps + 6 handcrafted
        # LightGBM is trained on these exact 9.

        # -----------------------------------------------------
        # Apply scaler from training
        # -----------------------------------------------------
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X, X.shape[1], None

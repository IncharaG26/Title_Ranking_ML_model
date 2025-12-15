# src/data_loader.py

import os
import pandas as pd
import json

# -------------------------
# Load ARXIV dataset
# -------------------------
def load_arxiv(folder):
    arxiv_path = os.path.join(folder, "arxiv_titles.xlsx")
    if os.path.exists(arxiv_path):
        df = pd.read_excel(arxiv_path)
        df = df.rename(columns={"Title": "title", "Abstract": "abstract"})
        df = df[["title", "abstract"]]
        return df
    return pd.DataFrame(columns=["title", "abstract"])


# -------------------------
# Load ICML parquet files
# -------------------------
def load_icml(folder):
    dfs = []
    for year in ["2021", "2022", "2023", "2024"]:
        p = os.path.join(folder, f"icml_{year}-00000-of-00001.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df = df.rename(columns={"title": "title", "abstract": "abstract"})
            df = df[["title", "abstract"]]
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["title", "abstract"])


# -------------------------
# Load S2ORC metadata & pdf_parses
# -------------------------
def load_s2orc(folder):
    meta_path = os.path.join(folder, "metadata", "sample.jsonl")
    parses_path = os.path.join(folder, "pdf_parses", "sample.jsonl")

    if not (os.path.exists(meta_path) and os.path.exists(parses_path)):
        return pd.DataFrame(columns=["title", "abstract"])

    meta_df = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if "title" in j and "abstract" in j:
                meta_df.append({"title": j["title"], "abstract": j.get("abstract", "")})

    return pd.DataFrame(meta_df)


# -------------------------
# MASTER DATASET LOADER  (NEW)
# -------------------------
def load_datasets(folder):
    print("→ Loading ArXiv")
    df1 = load_arxiv(folder)

    print("→ Loading ICML")
    df2 = load_icml(folder)

    print("→ Loading S2ORC")
    df3 = load_s2orc(folder)

    final_df = pd.concat([df1, df2, df3], ignore_index=True)

    final_df = final_df.dropna(subset=["title", "abstract"])
    final_df = final_df[final_df["title"] != ""]
    final_df = final_df[final_df["abstract"] != ""]

    return final_df.reset_index(drop=True)

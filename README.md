Scientific Paper Title Relevance Scoring using ArXiv Dataset

This project builds a Machine Learning model to evaluate how well a research paper title matches its abstract.
The model is trained using real scientific metadata from the ArXiv Kaggle dataset (Cornell University).

The system outputs:

A semantic similarity score (0.10 — 0.95)

A relevance category

Highly Relevant

Relevant

Partially Relevant

Low Relevance

A Tkinter-based GUI is provided for easy testing with custom titles and abstracts.

Dataset Information

Source: Kaggle (Cornell University — ArXiv Metadata)

Fields Used: Title, Abstract

Loading Method: Streaming JSON (memory-safe)

Rows Loaded Initially: 20,000

Rows Used for Training After Cleaning: ~15,000

Dataset Processing Steps
Step	Description
Load ArXiv metadata	Stream-first approach to avoid MemoryError
Clean text	Using simple_clean()
Remove duplicates	Based on title + abstract signature
Create semantic label	Cosine similarity using SBERT embeddings
Feature extraction	SBERT + TF-IDF + SVD fusion
Model used	LightGBM Regressor
Model Pipeline Flow


Score Classification
Score Range	Category	Meaning
≥ 0.80	Highly Relevant	Strong semantic alignment
0.60 – 0.79	Relevant	Good match
0.40 – 0.59	Partially Relevant	Some mismatch
< 0.40	Low Relevance	Weak or unrelated
Folder Structure
Title_Ranking_ML_model/
│
├── src/                          # Preprocessing & feature scripts
├── outputs/                      # Saved model + predictions
├── datasets/                     # ArXiv dataset (ignored in Git)
├── gui_app.py                    # GUI application
├── run_pipeline_final.py         # Main training script
├── bulk_test.py                  # Batch testing
├── model_test_lgb.py             # Model evaluation
├── requirements.txt
└── README.md

Installation & Setup
1️⃣ Create Virtual Environment
python -m venv venv


Activate:

Windows:

venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

Training the Model
python run_pipeline_final.py


Outputs stored in:

outputs/

Running the GUI
python gui_app.py


Enter:

Title

Abstract
→ Model will display score + category

Example Test Records (GUI Demo)

Deep Learning Approaches for Medical Image Segmentation

Score: 0.88

Category: Highly Relevant

Results

Successfully trained a model using real research data

Model generalizes well to unseen title–abstract pairs

GUI demonstrates real-time prediction

Future Improvements

Fine-tuning SBERT on ArXiv domain

Multi-category classification with expert labels

Deploy GUI as a web application (Streamlit)

Contributor

Inchara G

Scientific Paper Title Relevance Scoring (ArXiv Dataset)
📌 Overview

This project implements a Machine Learning system that evaluates how well a research paper title matches its abstract.
It predicts:

A relevance score (between 0.10 – 0.95)

A relevance category
(Highly Relevant, Relevant, Partially Relevant, Low Relevance)

A Python Tkinter GUI is included to test title-abstract pairs interactively.

🗄 Dataset Details
Property	Description
Source	Kaggle – Cornell University (arXiv Metadata)
Type	Scientific Research Paper Metadata
Fields Used	title, abstract
Total Raw Records	2.2 Million+
Loaded for Training	First 20,000 streamed JSON rows
Final Records after Cleaning	~15,000 unique title-abstract pairs
📌 Why ArXiv Dataset?

Real academic content (high-quality text)

Public domain metadata (free for projects)

Covers multiple scientific domains

Ideal for semantic similarity learning

🧹 Data Processing Steps
Step	Description
Streaming JSON load	Prevents MemoryError
Text cleaning	Removal of noise, special chars
Duplicate removal	Title+abstract signature
Score labeling	SBERT cosine similarity
Shuffle	Avoids ordering bias

Target score formula:

score = 0.10 + cosine_similarity * 0.85

🔍 Feature Engineering
Component	Purpose
SBERT Embeddings	Contextual semantic representation
TF-IDF Matrix	Keyword-based importance
SVD	Dimensionality reduction
FeatureFusionBuilder	Unified optimized feature set
🚀 Model Training
Model	LightGBM Regressor
Validation	5-Fold Cross Validation
Metrics	RMSE & Spearman Correlation
Output Range	0.10 → 0.95

Saved output files (in outputs/):

models/lgbm.joblib

scaler.joblib

feature_builder.joblib

predictions_lgbm.csv

pipeline_meta.json

🏷 Score Classification
Score Range	Category	Interpretation
≥ 0.80	Highly Relevant	Very strong match
0.60 – 0.79	Relevant	Good match
0.40 – 0.59	Partially Relevant	Acceptable match
< 0.40	Low Relevance	Weak or unrelated
🖥 GUI Application

A simple desktop GUI is provided using Tkinter.

Launch GUI:
python gui_app.py

GUI Output Example
Input Title	Score	Category
Deep Learning Approaches for Medical Image Segmentation	0.88	Highly Relevant
📁 Project Structure
Title_Ranking_ML_model/
│
├── src/                        # Preprocessing + feature tools
├── outputs/                    # Model artifacts & predictions
├── datasets/                   # ArXiv data (ignored in Git)
├── gui_app.py                  # GUI program
├── run_pipeline_final.py       # Full training pipeline
├── bulk_test.py                # Batch prediction script
├── model_test_lgb.py           # Model evaluation
├── requirements.txt
└── README.md

⚙️ Installation
pip install -r requirements.txt

(Optional) Create a Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate

▶️ Model Training
python run_pipeline_final.py


Output artifacts generated automatically inside /outputs.

📌 Future Improvements

Improve SBERT with domain-specific fine-tuning

Deploy model as a Web Application

Add explainability metrics for research evaluation

👤 Author

Inchara G
Computer Science Engineering StudentScientific Paper Title Relevance Scoring (ArXiv Dataset)
📌 Overview

This project implements a Machine Learning system that evaluates how well a research paper title matches its abstract.
It predicts:

A relevance score (between 0.10 – 0.95)

A relevance category
(Highly Relevant, Relevant, Partially Relevant, Low Relevance)

A Python Tkinter GUI is included to test title-abstract pairs interactively.

📂 System Architecture
flowchart TD
    A[Load ArXiv Dataset<br>Streaming JSON] --> B[Preprocess & Normalize Text<br>(simple_clean)]
    B --> C[Remove Duplicates<br>empty/null filtering]
    C --> D[Generate SBERT Embeddings<br>for title & abstract]
    D --> E[Compute Cosine Similarity<br>Target Score]
    E --> F[Feature Fusion<br>TF-IDF + SVD + SBERT]
    F --> G[Train Model<br>LightGBM + 5-Fold CV]
    G --> H[Save Artifacts<br>model, scaler, features, predictions]
    H --> I[Test via GUI<br>Score + Category Output]

🗄 Dataset Details
Property	Description
Source	Kaggle – Cornell University (arXiv Metadata)
Type	Scientific Research Paper Metadata
Fields Used	title, abstract
Total Raw Records	2.2 Million+
Loaded for Training	First 20,000 streamed JSON rows
Final Records after Cleaning	~15,000 unique title-abstract pairs
📌 Why ArXiv Dataset?

Real academic content (high-quality text)

Public domain metadata (free for projects)

Covers multiple scientific domains

Ideal for semantic similarity learning

🧹 Data Processing Steps
Step	Description
Streaming JSON load	Prevents MemoryError
Text cleaning	Removal of noise, special chars
Duplicate removal	Title+abstract signature
Score labeling	SBERT cosine similarity
Shuffle	Avoids ordering bias

Target score formula:

score = 0.10 + cosine_similarity * 0.85

🔍 Feature Engineering
Component	Purpose
SBERT Embeddings	Contextual semantic representation
TF-IDF Matrix	Keyword-based importance
SVD	Dimensionality reduction
FeatureFusionBuilder	Unified optimized feature set
🚀 Model Training
Model	LightGBM Regressor
Validation	5-Fold Cross Validation
Metrics	RMSE & Spearman Correlation
Output Range	0.10 → 0.95

Saved output files (in outputs/):

models/lgbm.joblib

scaler.joblib

feature_builder.joblib

predictions_lgbm.csv

pipeline_meta.json

🏷 Score Classification
Score Range	Category	Interpretation
≥ 0.80	Highly Relevant	Very strong match
0.60 – 0.79	Relevant	Good match
0.40 – 0.59	Partially Relevant	Acceptable match
< 0.40	Low Relevance	Weak or unrelated
🖥 GUI Application

A simple desktop GUI is provided using Tkinter.

Launch GUI:
python gui_app.py

GUI Output Example
Input Title	Score	Category
Deep Learning Approaches for Medical Image Segmentation	0.88	Highly Relevant
📁 Project Structure
Title_Ranking_ML_model/
│
├── src/                        # Preprocessing + feature tools
├── outputs/                    # Model artifacts & predictions
├── datasets/                   # ArXiv data (ignored in Git)
├── gui_app.py                  # GUI program
├── run_pipeline_final.py       # Full training pipeline
├── bulk_test.py                # Batch prediction script
├── model_test_lgb.py           # Model evaluation
├── requirements.txt
└── README.md

⚙️ Installation
pip install -r requirements.txt

(Optional) Create a Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate

▶️ Model Training
python run_pipeline_final.py


Output artifacts generated automatically inside /outputs.

📌 Future Improvements

Improve SBERT with domain-specific fine-tuning

Deploy model as a Web Application

Add explainability metrics for research evaluation

👤 Author

Inchara G
Computer Science Engineering StudentScientific Paper Title Relevance Scoring (ArXiv Dataset)
📌 Overview

This project implements a Machine Learning system that evaluates how well a research paper title matches its abstract.
It predicts:

A relevance score (between 0.10 – 0.95)

A relevance category
(Highly Relevant, Relevant, Partially Relevant, Low Relevance)

A Python Tkinter GUI is included to test title-abstract pairs interactively.

📂 System Architecture
flowchart TD
    A[Load ArXiv Dataset<br>Streaming JSON] --> B[Preprocess & Normalize Text<br>(simple_clean)]
    B --> C[Remove Duplicates<br>empty/null filtering]
    C --> D[Generate SBERT Embeddings<br>for title & abstract]
    D --> E[Compute Cosine Similarity<br>Target Score]
    E --> F[Feature Fusion<br>TF-IDF + SVD + SBERT]
    F --> G[Train Model<br>LightGBM + 5-Fold CV]
    G --> H[Save Artifacts<br>model, scaler, features, predictions]
    H --> I[Test via GUI<br>Score + Category Output]

🗄 Dataset Details
Property	Description
Source	Kaggle – Cornell University (arXiv Metadata)
Type	Scientific Research Paper Metadata
Fields Used	title, abstract
Total Raw Records	2.2 Million+
Loaded for Training	First 20,000 streamed JSON rows
Final Records after Cleaning	~15,000 unique title-abstract pairs
📌 Why ArXiv Dataset?

Real academic content (high-quality text)

Public domain metadata (free for projects)

Covers multiple scientific domains

Ideal for semantic similarity learning

🧹 Data Processing Steps
Step	Description
Streaming JSON load	Prevents MemoryError
Text cleaning	Removal of noise, special chars
Duplicate removal	Title+abstract signature
Score labeling	SBERT cosine similarity
Shuffle	Avoids ordering bias

Target score formula:

score = 0.10 + cosine_similarity * 0.85

🔍 Feature Engineering
Component	Purpose
SBERT Embeddings	Contextual semantic representation
TF-IDF Matrix	Keyword-based importance
SVD	Dimensionality reduction
FeatureFusionBuilder	Unified optimized feature set
🚀 Model Training
Model	LightGBM Regressor
Validation	5-Fold Cross Validation
Metrics	RMSE & Spearman Correlation
Output Range	0.10 → 0.95

Saved output files (in outputs/):

models/lgbm.joblib

scaler.joblib

feature_builder.joblib

predictions_lgbm.csv

pipeline_meta.json

🏷 Score Classification
Score Range	Category	Interpretation
≥ 0.80	Highly Relevant	Very strong match
0.60 – 0.79	Relevant	Good match
0.40 – 0.59	Partially Relevant	Acceptable match
< 0.40	Low Relevance	Weak or unrelated
🖥 GUI Application

A simple desktop GUI is provided using Tkinter.

Launch GUI:
python gui_app.py

GUI Output Example
Input Title	Score	Category
Deep Learning Approaches for Medical Image Segmentation	0.88	Highly Relevant
📁 Project Structure
Title_Ranking_ML_model/
│
├── src/                        # Preprocessing + feature tools
├── outputs/                    # Model artifacts & predictions
├── datasets/                   # ArXiv data (ignored in Git)
├── gui_app.py                  # GUI program
├── run_pipeline_final.py       # Full training pipeline
├── bulk_test.py                # Batch prediction script
├── model_test_lgb.py           # Model evaluation
├── requirements.txt
└── README.md

⚙️ Installation
pip install -r requirements.txt

(Optional) Create a Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate

▶️ Model Training
python run_pipeline_final.py


Output artifacts generated automatically inside /outputs.

📌 Future Improvements

Improve SBERT with domain-specific fine-tuning

Deploy model as a Web Application

Add explainability metrics for research evaluation

👤 Contributor

Inchara G

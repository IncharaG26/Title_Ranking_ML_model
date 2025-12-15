# Title Ranking ML Model

A machine learning pipeline for ranking and scoring academic paper titles based on their relevance and quality. This project uses LightGBM with SBERT (Sentence Transformers) embeddings and custom feature fusion to predict title-abstract matching scores.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Testing with GUI](#2-testing-with-gui)
  - [3. Bulk Evaluation](#3-bulk-evaluation)
  - [4. Model Testing](#4-model-testing)
- [Datasets](#datasets)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

This project implements a complete machine learning pipeline for academic title ranking that:

- **Preprocesses** text data (cleaning, deduplication)
- **Generates Features** using:
  - SBERT embeddings (semantic similarity)
  - Lexical features (token overlap, length ratio)
  - Fusion-based scoring
- **Trains** a LightGBM regression model with K-Fold cross-validation
- **Evaluates** model performance with multiple metrics
- **Provides** both GUI and CLI interfaces for predictions

## 📦 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows/Mac/Linux
- **RAM**: Minimum 8GB (16GB recommended for SBERT model)
- **Disk Space**: ~5GB (for SBERT model and datasets)

### Required Software

- Git (for cloning the repository)
- Python package manager (pip)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KavanaN12/Title_Ranking_ML_model.git
cd Title_Ranking_ML_model/title_ranking_project
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- `numpy`, `pandas` - Data processing
- `scikit-learn` - Machine learning utilities
- `lightgbm` - LightGBM model
- `sentence-transformers` - SBERT embeddings
- `nltk`, `scipy` - NLP utilities
- `matplotlib` - Visualization
- `streamlit` - Web interface
- And other dependencies

### 4. Download Required NLP Data

After installation, download NLTK data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 📁 Project Structure

```
title_ranking_project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── run_pipeline_final.py          # Main training pipeline
├── gui_app.py                     # Tkinter GUI for predictions
├── bulk_test.py                   # Bulk evaluation on test dataset
├── model_test_lgb.py              # Detailed model testing
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Text preprocessing functions
│   ├── features_fusion.py         # Feature extraction and fusion
│   ├── models.py                  # Model definitions
│   ├── train_eval.py              # Training and evaluation utilities
│   └── utils.py                   # Helper functions
├── outputs/                       # Generated artifacts (after training)
│   ├── models/
│   │   └── lgbm.joblib           # Trained LightGBM model
│   ├── feature_builder.joblib    # Feature builder object
│   ├── scaler.joblib             # StandardScaler for features
│   ├── target_stats.json         # Target variable statistics
│   ├── predictions_lgbm.csv      # Training predictions
│   ├── pipeline_meta.json        # Pipeline metadata
│   ├── bulk_test_results/        # Bulk evaluation results
│   └── model_test_plots/         # Test visualization plots
└── datasets/                      # Data directory (symlink or copy)
    ├── train_real_world_dataset_10000.csv  # Training dataset
    └── real_world_dataset_2000_cleaned.csv # Test/evaluation dataset
```

## 💻 Usage

### 1. Training the Model

To train the model from scratch using the training dataset:

```bash
python run_pipeline_final.py
```

**What happens:**
- Loads training data from `datasets/train_real_world_dataset_10000.csv`
- Preprocesses and cleans text
- Builds SBERT embeddings and fusion features
- Trains LightGBM with 5-Fold cross-validation
- Saves all artifacts to `outputs/`
- Generates initial predictions on training data

**Expected output:**
- `outputs/models/lgbm.joblib` - Trained model
- `outputs/feature_builder.joblib` - Feature builder
- `outputs/scaler.joblib` - Feature scaler
- `outputs/target_stats.json` - Target statistics
- `outputs/predictions_lgbm.csv` - Training predictions

**Estimated time:** 10-30 minutes (depending on hardware)

### 2. Testing with GUI

Launch the interactive GUI for single predictions:

```bash
python gui_app.py
```

**Features:**
- Enter title and abstract manually
- Get instant predictions with confidence scores
- Category mapping (Excellent, Strong, Moderate, Weak, NoMatch)
- Simple, user-friendly interface

**Requirements:**
- Model must be trained first (run `run_pipeline_final.py`)

### 3. Bulk Evaluation

Run batch predictions and evaluation on the test dataset:

```bash
python bulk_test.py
```

**What happens:**
- Loads test dataset from `datasets/real_world_dataset_2000_cleaned.csv`
- Generates predictions for all records
- Computes evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Spearman/Pearson Correlation
- Creates confusion matrix visualization
- Generates category distribution plots

**Output:**
- `outputs/bulk_test_results/metrics.json` - Performance metrics
- `outputs/bulk_test_results/predictions_bulk.csv` - Bulk predictions
- `outputs/bulk_test_results/plots/` - Visualization plots

**Estimated time:** 2-5 minutes

### 4. Detailed Model Testing

Generate comprehensive test report with detailed analysis:

```bash
python model_test_lgb.py
```

**What happens:**
- Loads trained model and artifacts
- Computes detailed performance metrics
- Generates individual feature importance plots
- Creates prediction distribution plots
- Produces residual analysis

**Output:**
- `outputs/model_test_plots/` - Detailed test plots
- Console output with performance summary

## 📊 Datasets

### Training Dataset

**Location:** `datasets/train_real_world_dataset_10000.csv`

- **Size:** 10,000 records
- **Source:** CrossRef / Real-world academic papers
- **Format:** CSV with columns:
  - `title` - Paper title
  - `abstract` - Paper abstract
  - `expected` - Target relevance score (0-1)

### Test/Evaluation Dataset

**Location:** `datasets/real_world_dataset_2000_cleaned.csv`

- **Size:** 2,000 records
- **Source:** Real-world academic papers (non-overlapping with training)
- **Format:** Same as training dataset
- **Usage:** Bulk evaluation and model validation

### Required Dataset Columns

Both datasets must have:
- `title` (string) - Paper title
- `abstract` (string) - Paper abstract
- `expected` (float) - Target score (range 0-1)

## 📤 Outputs

After training and evaluation, the following artifacts are generated:

### Models & Artifacts

| File | Description |
|------|-------------|
| `models/lgbm.joblib` | Trained LightGBM model |
| `feature_builder.joblib` | FeatureFusionBuilder object |
| `scaler.joblib` | StandardScaler for normalization |
| `target_stats.json` | Mean/std of target variable |
| `pipeline_meta.json` | Pipeline metadata and config |

### Predictions & Metrics

| File | Description |
|------|-------------|
| `predictions_lgbm.csv` | Training set predictions |
| `bulk_test_results/predictions_bulk.csv` | Test set predictions |
| `bulk_test_results/metrics.json` | Performance metrics |

### Visualizations

| File | Description |
|------|-------------|
| `bulk_test_results/plots/` | Test set plots (distribution, confusion matrix, etc.) |
| `model_test_plots/` | Detailed model analysis plots |

## ⚙️ Configuration

### Main Configuration Variables

Located in `run_pipeline_final.py`:

```python
DATASET_FOLDER = "D:/aimlTextPr/datasets"  # Dataset location
CROSSREF_TRAIN_PATH = "datasets/train_real_world_dataset_10000.csv"
EVAL_TEST_PATH = "datasets/real_world_dataset_2000_cleaned.csv"
OUT_DIR = "outputs"
SBERT_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
SEED = 42
N_SPLITS = 5  # K-Fold splits
```

### To Customize

1. **Change training dataset:** Modify `CROSSREF_TRAIN_PATH`
2. **Change test dataset:** Modify `EVAL_TEST_PATH`
3. **Change SBERT model:** Modify `SBERT_MODEL` to another HuggingFace model
4. **Adjust K-Fold splits:** Change `N_SPLITS` value
5. **Change random seed:** Modify `SEED` for reproducibility

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sentence_transformers'"

**Solution:**
```bash
pip install sentence-transformers --upgrade
```

### Issue: "CUDA out of memory" or slow processing

**Solution:**
- Reduce batch size in feature builder
- Use smaller SBERT model:
  ```python
  SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
  ```
- Ensure sufficient RAM available

### Issue: "Dataset file not found"

**Solution:**
- Verify dataset paths in script match your system
- Update absolute paths in configuration
- Ensure datasets directory exists with required CSV files

### Issue: GUI window doesn't open

**Solution:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- On Linux, may need: `sudo apt-get install python3-tk`
- Try running from command line to see error messages

### Issue: "Model not found" when running GUI or testing

**Solution:**
- Train the model first: `python run_pipeline_final.py`
- Wait for training to complete and artifacts to be saved
- Check `outputs/` folder for model files

### Issue: Low model performance

**Consider:**
- Verify dataset quality and format
- Check feature engineering settings in `features_fusion.py`
- Increase training data size
- Adjust model hyperparameters in `run_pipeline_final.py`
- Review data preprocessing in `preprocess.py`

## 📚 Key Components

### Feature Fusion Builder (`src/features_fusion.py`)

Generates comprehensive features:
- **SBERT Embeddings:** Semantic similarity between title and abstract
- **Lexical Features:** Token overlap, length ratios, BM25 scores
- **Fusion Score:** Combined metric from all feature sources

### Preprocessing (`src/preprocess.py`)

Text cleaning:
- Lowercase conversion
- Special character removal
- Whitespace normalization
- Deduplication

### Model Training (`src/train_eval.py`)

- K-Fold cross-validation
- StandardScaler normalization
- LightGBM regression with early stopping
- Multiple evaluation metrics

## 📝 License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## 👤 Author

**Kavana N**

GitHub: [@KavanaN12](https://github.com/KavanaN12)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues, questions, or suggestions, please:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

## 🔗 Related Resources

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)

---

**Last Updated:** December 2025

**Status:** Active Development
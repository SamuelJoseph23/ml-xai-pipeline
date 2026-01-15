# ML Model & XAI Pipeline

This repository contains an end-to-end Machine Learning pipeline for tabular data, featuring robust preprocessing, multi-model training, and advanced Explainable AI (XAI) reports.

## Core Features

- **Categorical Support**: Automatically detects and encodes categorical features using `OneHotEncoder`.
- **Data Leakage Prevention**: Automatically identifies and drops source columns used to derive target labels to ensure realistic evaluation.
- **Multi-Class Robustness**: Handles binary and multi-class classification with weighted performance metrics (Precision, Recall, F1) and One-vs-Rest ROC-AUC.
- **Interactive Explanations**: Generates HTML-based ELI5 reports for global feature weights and local prediction debugging.
- **Dynamic Configuration**: Fully parameterized scripts support custom input paths, target columns, and tuning parameters via CLI.

---

## Installation

Create a virtual environment and install the required dependencies:

```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install pandas scikit-learn matplotlib eli5 numpy
```

---

## Usage Workflow

### Step 1: Data Preprocessing
Clean and scale your raw dataset. This script handles missing values (median/most-frequent imputation), standardizes numeric features, and One-Hot encodes categorical data.

```bash
python preprocess.py --input device1_top_20_features.csv --output preprocessed_data.csv --label label
```

- **Default Input**: `device1_top_20_features.csv`
- **Output**: Scaled and encoded `.csv` ready for training.

### Step 2: Model Training & XAI
Train Decision Tree, Random Forest, and Neural Network (MLP) models.

```bash
python xai.py --input preprocessed_data.csv --source_col HH_L5_pcc
```

- **Arguments**:
  - `--input`: Path to the file from Step 1.
  - `--source_col`: The feature used to derive targets (if any). This column will be dropped to prevent **Data Leakage**.
  - `--num_examples`: Number of local prediction explanations to generate (default: 5).

---

## Outputs

1.  **Performance Metrics**: Detailed console output including Accuracy, Weighted F1-Score, and Weighted OVR ROC-AUC.
2.  **`feature_importances.png`**: A high-resolution comparison plot of the top 12 features for each model.
3.  **`eli5_explanation_<Model>.html`**: Interactive HTML reports containing:
    - **Global Summary**: Feature weights and decision logic.
    - **Local Explanations**: Breakdown of individual predictions for specific samples.

## Performance Benchmark (`device1` dataset)

| Model | Accuracy | F1 (Weighted) | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Decision Tree** | ~0.81 | ~0.81 | ~0.94 |
| **Random Forest** | ~0.82 | ~0.82 | ~0.97 |
| **Neural Network** | ~0.83 | ~0.83 | ~0.97 |

---

## Notes
- Make sure to use the provided `venv` for all executions to ensure dependency compatibility.
- For multi-class datasets, metrics are calculated using **Weighted Averages** to account for class imbalance.

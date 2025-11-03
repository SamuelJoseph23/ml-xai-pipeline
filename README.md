# ML Model Pipeline

This repository contains two core files:
- **preprocess.py**: Preprocesses your raw dataset (handles missing values, scaling, label management).
- **xai.py**: Trains machine learning models on preprocessed data, evaluates performance, and generates feature importance and explainability reports.

---

## Requirements
pip install pandas scikit-learn matplotlib eli5 numpy


---

## Step 1: Data Preprocessing

Ensure your raw data CSV (e.g., `device1top20features.csv`) is in the same directory. Edit `preprocess.py` to specify your label column if it is not named "label".

Run preprocessing:
python preprocess.py


- **Input:** `device1top20features.csv`
- **Output:** `preprocesseddevice1features.csv`

---

## Step 2: Model Training, Evaluation & Explainability

After preprocessing, run the modeling script:
python xai.py


- **Input:** `preprocesseddevice1features.csv` (output of Step 1)
- **Outputs:**
  - **Performance metrics** (accuracy, precision) for Decision Tree, Random Forest, Neural Network (printed to console)
  - **Feature importances plot:** `featureimportances.png`
  - **Model explainability reports:** HTML files (`eli5explanation<MODEL_NAME>.html`) for each trained model, containing interpretable weights/rules and example prediction explanations

---

## Notes

- Check the top of both scripts if you need to adjust filenames or parameters.
- The HTML explanation files are viewable in any browser for human-readable model insights.
- Make sure Python 3.x is installed.

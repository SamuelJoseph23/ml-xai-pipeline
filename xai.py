import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import eli5
from eli5.formatters import format_as_html
import sys

# --- Data Loading ---
CSV_FILE = 'preprocessed_device1_features.csv'
TARGET_COL = 'HH_L5_pcc'
NUM_EXAMPLES = 5

try:
    df = pd.read_csv(CSV_FILE)
    print("Loaded existing data.")
except FileNotFoundError:
    print(f"File {CSV_FILE} not found.")
    sys.exit(1)

if TARGET_COL not in df.columns:
    print(f"Target column '{TARGET_COL}' not found in dataset.")
    sys.exit(1)

df['target'] = (df[TARGET_COL] > df[TARGET_COL].median()).astype(int)

X = df.drop(['target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Model Definitions ---
models = {
    'Decision Tree': DecisionTreeClassifier(
        random_state=42, max_depth=6, min_samples_split=5),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    'Neural Network (MLP)': MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42,
        early_stopping=True)
}

results = {}

# --- Model Training & Evaluation ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

# --- Print Performance Metrics ---
print("\nModel Performance Metrics:")
print("="*65)
print(f"{'Model':<25}{'Accuracy':>12}{'Precision':>12}{'Recall':>10}{'F1':>10}")
print("-"*65)
for model_name, metrics in results.items():
    print(f"{model_name:<25}{metrics['accuracy']:12.6f}{metrics['precision']:12.6f}"
          f"{metrics['recall']:10.6f}{metrics['f1']:10.6f}")
print("="*65)

# --- ELI5 Explanations ---
for name, model in models.items():
    if name == 'Decision Tree':
        tree_rules = export_text(model, feature_names=list(X.columns), max_depth=3)
        weight_html = f"<pre>{tree_rules}</pre>"
    else:
        weight_explanation = eli5.explain_weights(
            model, feature_names=list(X.columns), top=10)
        weight_html = format_as_html(weight_explanation)

    explanations = []
    for i in range(min(NUM_EXAMPLES, len(X_test))):
        explanation = eli5.explain_prediction(
            model, X_test.iloc[i], feature_names=list(X.columns))
        explanations.append(format_as_html(explanation))

    file_name = f'eli5_explanation_{name.replace(" ", "_").replace("(", "").replace(")", "")}.html'
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            full_html = f"""
            <html>
            <head>
                <title>ELI5 Explanation for {name}</title>
                <style>
                    body {{font-family: Arial, sans-serif; margin: 20px;}}
                    .eli5-explanation {{margin-bottom: 40px;}}
                </style>
            </head>
            <body>
                <h1>ELI5 Explanation for {name}</h1>
                <h2>Feature Importances / Decision Summary</h2>
                <div class="eli5-explanation">{weight_html}</div>
                <h2>Example Predictions</h2>
                {"".join(f'<div class="eli5-explanation"><h3>Prediction {i+1}</h3>{expl}</div>'
                         for i, expl in enumerate(explanations))}
            </body>
            </html>
            """
            f.write(full_html)
    except Exception as e:
        print(f"Failed to write ELI5 explanation HTML for {name}: {e}")

# --- Feature Importances ---
feature_importances = {}
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        perm_result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_result['importances_mean']

    fi_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    feature_importances[name] = fi_df

# --- Plot Feature Importances ---
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for i, (model_name, fi_df) in enumerate(feature_importances.items()):
    ax = axes[i]
    top_features = fi_df.head(10)
    ax.barh(top_features['feature'], top_features['importance'], color='skyblue')
    ax.set_title(model_name)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, np.max(top_features['importance']) * 1.1)

plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nHTML explanation files have been generated:")
for name in models.keys():
    file_name = f'eli5_explanation_{name.replace(" ", "_").replace("(", "").replace(")", "")}.html'
    print(f"- {file_name}")

sys.exit(0)

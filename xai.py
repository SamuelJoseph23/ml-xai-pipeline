import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import eli5
from eli5.formatters import format_as_html

def train_and_evaluate(args):
    try:
        df = pd.read_csv(args.input)
        print(f"[INFO] Loaded existing data from {args.input}.")
    except FileNotFoundError:
        print(f"[ERROR] File {args.input} not found.")
        sys.exit(1)

    # Derived 'label' if it doesn't exist, using the specified source column
    if 'label' not in df.columns:
        if args.source_col not in df.columns:
            print(f"[ERROR] Source column '{args.source_col}' for target not found.")
            sys.exit(1)
        print(f"[INFO] Deriving 'target' from '{args.source_col}' (median split).")
        df['target'] = (df[args.source_col] > df[args.source_col].median()).astype(int)
        y = df['target']
        # FIX DATA LEAKAGE: Drop the column used to derive the target
        X = df.drop(['target', args.source_col], axis=1)
    else:
        print("[INFO] Using existing 'label' column.")
        y = df['label']
        X = df.drop(['label'], axis=1)
        if args.source_col in X.columns:
            print(f"[WARNING] Potential leakage: Source col '{args.source_col}' found in features. Dropping it.")
            X = X.drop([args.source_col], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"[INFO] Training with {X_train.shape[1]} features.")

    # --- Enhanced Model Definitions ---
    models = {
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=8, min_samples_split=10),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1),
        'Neural Network (MLP)': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            alpha=0.0001,
            max_iter=500,
            random_state=42,
            early_stopping=True)
    }

    results = {}
    trained_models = {}

    # --- Training & Evaluation ---
    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate ROC-AUC with multi-class support
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                # multi_class='ovr' handles any number of classes
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            else:
                roc_auc = 0.0 # Placeholder if no proba
        except Exception as e:
            print(f"[WARNING] Could not calculate ROC-AUC for {name}: {e}")
            roc_auc = 0.0
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc
        }
        trained_models[name] = model

    # --- Print Performance Metrics ---
    print("\nModel Performance Metrics (Weighted Averages):")
    print("="*90)
    print(f"{'Model':<25}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'ROC-AUC':>12}")
    print("-"*90)
    for model_name, metrics in results.items():
        print(f"{model_name:<25}{metrics['accuracy']:12.4f}{metrics['precision']:12.4f}"
              f"{metrics['recall']:12.4f}{metrics['f1']:12.4f}{metrics['roc_auc']:12.4f}")
    print("="*90)


    # --- ELI5 Explanations ---
    for name, model in trained_models.items():
        print(f"[INFO] Generating ELI5 report for {name}...")
        if name == 'Decision Tree':
            tree_rules = export_text(model, feature_names=list(X.columns), max_depth=3)
            weight_html = f"<pre>{tree_rules}</pre>"
        else:
            weight_explanation = eli5.explain_weights(
                model, feature_names=list(X.columns), top=15)
            weight_html = format_as_html(weight_explanation)

        explanations = []
        for i in range(min(args.num_examples, len(X_test))):
            explanation = eli5.explain_prediction(
                model, X_test.iloc[i], feature_names=list(X.columns))
            explanations.append(format_as_html(explanation))

        file_name = f'eli5_explanation_{name.replace(" ", "_").replace("(", "").replace(")", "")}.html'
        full_html = f"""
        <html>
        <head>
            <title>ELI5 Explanation for {name}</title>
            <style>
                body {{font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9;}}
                .container {{max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}}
                h1 {{color: #333; border-bottom: 2px solid #3498db; padding-bottom: 10px;}}
                h2 {{color: #2c3e50; margin-top: 30px;}}
                .eli5-explanation {{margin-bottom: 40px; overflow-x: auto;}}
                pre {{background: #eee; padding: 15px; border-radius: 4px;}}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ELI5 Explanation for {name}</h1>
                <h2>Global Feature Importance / Decision Summary</h2>
                <div class="eli5-explanation">{weight_html}</div>
                <hr>
                <h2>Local Example Predictions</h2>
                {"".join(f'<div class="eli5-explanation"><h3>Example Prediction {i+1}</h3>{expl}</div>'
                         for i, expl in enumerate(explanations))}
            </div>
        </body>
        </html>
        """
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(full_html)

    # --- Feature Importances Plot ---
    print("[INFO] Plotting feature importances...")
    feature_importances = {}
    for name, model in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            perm_result = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42)
            importances = perm_result['importances_mean']

        fi_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        feature_importances[name] = fi_df

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for i, (model_name, fi_df) in enumerate(feature_importances.items()):
        ax = axes[i]
        top_features = fi_df.head(12).sort_values('importance', ascending=True)
        ax.barh(top_features['feature'], top_features['importance'], color=plt.cm.viridis(np.linspace(0, 1, 12)))
        ax.set_title(f"Top Features: {model_name}", fontsize=14)
        ax.set_xlabel('Importance Score')
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Reports and plots generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models and generate XAI reports.")
    parser.add_argument("--input", default="preprocessed_device1_features.csv", help="Input preprocessed CSV")
    parser.add_argument("--source_col", default="HH_L5_pcc", help="Original col used to derive target (to drop)")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of example predictions to explain")
    args = parser.parse_args()
    train_and_evaluate(args)


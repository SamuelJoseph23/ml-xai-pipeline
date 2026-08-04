import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
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
        y = df['label'].astype(int)
        X = df.drop(['label'], axis=1)
        if args.source_col in X.columns:
            print(f"[WARNING] Potential leakage: Source col '{args.source_col}' found in features. Dropping it.")
            X = X.drop([args.source_col], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"[INFO] Training with {X_train.shape[1]} features.")

    # --- Individual Model Definitions ---
    dt  = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_split=10)
    rf  = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
    ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                        alpha=0.0001, max_iter=500, random_state=42, early_stopping=True)

    # --- Voting Ensemble (soft voting uses predicted probabilities) ---
    ensemble = VotingClassifier(
        estimators=[
            ('Decision Tree', dt),
            ('Random Forest', rf),
            ('AdaBoost',      ada),
            ('SVM',           svm),
            ('MLP',           mlp),
        ],
        voting='soft',
        n_jobs=-1
    )

    models = {
        'Decision Tree':        dt,
        'Random Forest':        rf,
        'AdaBoost':             ada,
        'SVM':                  svm,
        'Neural Network (MLP)': mlp,
        'Voting Ensemble':      ensemble,
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
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            else:
                roc_auc = 0.0
        except Exception as e:
            print(f"[WARNING] Could not calculate ROC-AUC for {name}: {e}")
            roc_auc = 0.0

        results[name] = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall':    recall_score(y_test, y_pred, average='weighted'),
            'f1':        f1_score(y_test, y_pred, average='weighted'),
            'roc_auc':   roc_auc,
        }
        trained_models[name] = model

    # --- Print Performance Metrics ---
    print("\nModel Performance Metrics (Weighted Averages):")
    print("=" * 100)
    print(f"{'Model':<25}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'ROC-AUC':>12}")
    print("-" * 100)
    for model_name, metrics in results.items():
        marker = " <-- ENSEMBLE" if model_name == 'Voting Ensemble' else ""
        print(f"{model_name:<25}{metrics['accuracy']:12.4f}{metrics['precision']:12.4f}"
              f"{metrics['recall']:12.4f}{metrics['f1']:12.4f}{metrics['roc_auc']:12.4f}{marker}")
    print("=" * 100)

    # --- Model Comparison Bar Chart ---
    print("[INFO] Generating model comparison chart...")
    metrics_to_plot = ['accuracy', 'precision', 'f1', 'roc_auc']
    metric_labels   = ['Accuracy', 'Precision', 'F1-Score', 'ROC-AUC']
    model_names     = list(results.keys())
    n_metrics       = len(metrics_to_plot)
    n_models        = len(model_names)
    x = np.arange(n_metrics)
    bar_width = 0.12

    # Highlight the Voting Ensemble bar in gold, others use tab10 palette
    palette = plt.cm.tab10(np.linspace(0, 0.85, n_models))
    ensemble_idx = model_names.index('Voting Ensemble')

    fig_cmp, ax_cmp = plt.subplots(figsize=(14, 7))
    for i, model_name in enumerate(model_names):
        offsets = x + (i - n_models / 2 + 0.5) * bar_width
        values  = [results[model_name][m] for m in metrics_to_plot]
        color   = '#FFB300' if model_name == 'Voting Ensemble' else palette[i]
        edge    = 'black'   if model_name == 'Voting Ensemble' else 'none'
        lw      = 1.5       if model_name == 'Voting Ensemble' else 0
        bars = ax_cmp.bar(offsets, values, width=bar_width, label=model_name,
                          color=color, edgecolor=edge, linewidth=lw, alpha=0.9)
        # Annotate bars with values
        for bar, val in zip(bars, values):
            ax_cmp.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=6.5, rotation=45)

    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(metric_labels, fontsize=12)
    ax_cmp.set_ylim(0.8, 1.02)
    ax_cmp.set_ylabel('Score', fontsize=12)
    ax_cmp.set_title('Model Comparison: Accuracy, Precision, F1 & ROC-AUC\n(Voting Ensemble highlighted in gold)',
                     fontsize=13, fontweight='bold')
    ax_cmp.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_cmp.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_cmp.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] model_comparison.png saved.")

    # --- ELI5 / Fallback Explanations (individual models only) ---
    # ELI5 natively supports: Decision Tree, Random Forest (global+local)
    # AdaBoost: global works, local explain_prediction NOT supported -> fallback table
    # SVM (rbf), MLP: neither global nor local supported -> permutation importance + fallback table
    ELI5_GLOBAL_OK  = {'Decision Tree', 'Random Forest', 'AdaBoost'}
    ELI5_LOCAL_OK   = {'Decision Tree', 'Random Forest'}

    def _perm_importance_html(model, feat_names, X_t, y_t):
        """Compute permutation importance and return a styled HTML table."""
        perm = permutation_importance(model, X_t, y_t, n_repeats=5, random_state=42, n_jobs=-1)
        fi = sorted(zip(feat_names, perm.importances_mean), key=lambda x: -x[1])[:15]
        rows = "".join(
            f"<tr><td style='padding:4px 12px;text-align:right;'>{v:.4f}</td>"
            f"<td style='padding:4px 12px;'>{f}</td></tr>"
            for f, v in fi
        )
        return (
            "<p><b>Permutation Importance</b> (top 15 features by mean accuracy drop):</p>"
            "<table style='border-collapse:collapse;'>"
            "<thead><tr><th style='padding:4px 12px;'>Importance</th>"
            "<th style='padding:4px 12px;text-align:left;'>Feature</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    def _local_fallback_html(model, sample, feat_names):
        """Return a styled table of feature values + prediction for unsupported models."""
        pred_class = model.predict(sample.values.reshape(1, -1))[0]
        prob_str = ""
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(sample.values.reshape(1, -1))[0]
            prob_str = " | ".join(f"Class {i}: {p:.3f}" for i, p in enumerate(probs))
        rows = "".join(
            f"<tr><td style='padding:3px 10px;'>{f}</td>"
            f"<td style='padding:3px 10px;text-align:right;'>{sample[f]:.4f}</td></tr>"
            for f in feat_names
        )
        return (
            f"<p><b>Predicted class:</b> {pred_class}"
            + (f" &nbsp;|&nbsp; <b>Probabilities:</b> {prob_str}" if prob_str else "")
            + "</p>"
            "<table style='border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr><th style='padding:3px 10px;text-align:left;'>Feature</th>"
            "<th style='padding:3px 10px;'>Value</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    eli5_models = {k: v for k, v in trained_models.items() if k != 'Voting Ensemble'}
    feat_names = list(X.columns)

    for name, model in eli5_models.items():
        print(f"[INFO] Generating explanation report for {name}...")

        # --- Global importance ---
        if name == 'Decision Tree':
            tree_rules = export_text(model, feature_names=feat_names, max_depth=3)
            weight_html = f"<pre>{tree_rules}</pre>"
        elif name in ELI5_GLOBAL_OK:
            weight_explanation = eli5.explain_weights(model, feature_names=feat_names, top=15)
            weight_html = format_as_html(weight_explanation)
        else:
            # SVM (rbf), MLP -> permutation importance
            weight_html = _perm_importance_html(model, feat_names, X_test, y_test)

        # --- Local predictions ---
        explanations = []
        for i in range(min(args.num_examples, len(X_test))):
            sample = X_test.iloc[i]
            if name in ELI5_LOCAL_OK:
                expl = eli5.explain_prediction(model, sample, feature_names=feat_names)
                explanations.append(format_as_html(expl))
            else:
                # AdaBoost, SVM, MLP -> feature value table fallback
                explanations.append(_local_fallback_html(model, sample, feat_names))

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        file_name = f'eli5_explanation_{safe_name}.html'
        full_html = f"""
        <html>
        <head>
            <title>XAI Explanation for {name}</title>
            <style>
                body {{font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9;}}
                .container {{max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}}
                h1 {{color: #333; border-bottom: 2px solid #3498db; padding-bottom: 10px;}}
                h2 {{color: #2c3e50; margin-top: 30px;}}
                .eli5-explanation {{margin-bottom: 40px; overflow-x: auto;}}
                pre {{background: #eee; padding: 15px; border-radius: 4px;}}
                table {{border-collapse: collapse;}}
                th, td {{border: 1px solid #ddd; padding: 4px 10px;}}
                th {{background: #f0f0f0;}}
                tr:hover {{background-color: #f9f9f9;}}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>XAI Explanation for {name}</h1>
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

    # --- Feature Importances Plot (all 6 models) ---
    print("[INFO] Plotting feature importances...")
    feature_importances = {}
    for name, model in trained_models.items():
        # For VotingClassifier, use RF sub-estimator importances as a proxy
        if name == 'Voting Ensemble':
            rf_sub = model.named_estimators_['Random Forest']
            importances = rf_sub.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            perm_result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            importances = perm_result['importances_mean']

        fi_df = pd.DataFrame({
            'feature':    X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        feature_importances[name] = fi_df

    n_models = len(feature_importances)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))
    colors = plt.cm.tab10(np.linspace(0, 0.9, 12))

    for i, (model_name, fi_df) in enumerate(feature_importances.items()):
        ax = axes[i]
        top_features = fi_df.head(12).sort_values('importance', ascending=True)
        bar_colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax.barh(top_features['feature'], top_features['importance'], color=bar_colors)
        label = f"Top Features:\n{model_name}" + (" (Ensemble)" if model_name == 'Voting Ensemble' else "")
        ax.set_title(label, fontsize=11, fontweight='bold')
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

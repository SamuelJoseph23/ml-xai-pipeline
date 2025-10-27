import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.inspection import permutation_importance
import eli5
from eli5.formatters import format_as_html
import sys

# Load data
try:
    df = pd.read_csv('preprocessed_device1_features.csv')
    print("Loaded existing data")
except FileNotFoundError:
    print("The file was not found.")
    sys.exit(1)

# Create target variable
df['target'] = (df['HH_L5_pcc'] > df['HH_L5_pcc'].median()).astype(int)

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42),
    'Neural Network (MLP)': MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=100,
        random_state=42,
        early_stopping=True
    )
}

# Train models and evaluate performance
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    results[name] = (accuracy, precision)

# Print model performance metrics
print("\nModel Performance Metrics:")
print("="*50)
print(f"{'Model':<25}{'Accuracy':>12}{'Precision':>12}")
print("-"*50)
for model_name, (acc, prec) in results.items():
    print(f"{model_name:<25}{acc:12.6f}{prec:12.6f}")
print("="*50)

# Generate ELI5 explanations
eli5_explanations = {}
for name, model in models.items():
    # Feature importance explanation
    if name == 'Decision Tree':
        # Use a cleaner textual rule summary
        tree_rules = export_text(model, feature_names=X.columns.tolist(), max_depth=3)
        weight_html = f"<pre>{tree_rules}</pre>"
    else:
        weight_explanation = eli5.explain_weights(
            model,
            feature_names=X.columns.tolist(),
            top=None
        )
        weight_html = format_as_html(weight_explanation)

    # Explain a few predictions
    explanations = []
    for i in range(5):  # Explain first 5 predictions
        explanation = eli5.explain_prediction(
            model, 
            X_test.iloc[i], 
            feature_names=X.columns.tolist(),
            top=None
        )
        explanations.append(format_as_html(explanation))

    eli5_explanations[name] = {
        'weight_html': weight_html,
        'prediction_htmls': explanations
    }

    # Save explanations to HTML files
    try:
        with open(f'eli5_explanation_{name}.html', 'w', encoding='utf-8') as f:
            full_html = f"""
            <html>
            <head>
                <title>ELI5 Explanation for {name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    .eli5-explanation {{ margin-bottom: 40px; }}
                </style>
            </head>
            <body>
                <h1>ELI5 Explanation for {name} Model</h1>

                <h2>Feature Importances / Decision Summary</h2>
                <div class="eli5-explanation">{weight_html}</div>

                <h2>Example Predictions</h2>
                {"".join(f'<div class="eli5-explanation"><h3>Prediction {i+1}</h3>{html}</div>' 
                        for i, html in enumerate(explanations))}
            </body>
            </html>
            """
            f.write(full_html)
    except Exception as e:
        print(f"Failed to write ELI5 explanation HTML for {name}: {e}")

# Calculate feature importances
feature_importances = {}
for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest']:
        importances = model.feature_importances_
    else:
        importances = permutation_importance(
            model, X_test, y_test, 
            n_repeats=10, 
            random_state=42
        )['importances'].mean(axis=1)

    fi_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    feature_importances[name] = fi_df

# Plot feature importances
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
titles = ['Decision Tree', 'Random Forest', 'Neural Network (MLP)']

for i, (model_name, fi_df) in enumerate(feature_importances.items()):
    ax = axes[i]
    top_features = fi_df.head(10)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_title(titles[i])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)

plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
plt.close()

# Optional: Print summary
print("\nHTML explanation files have been generated:")
for name in models.keys():
    print(f"- eli5_explanation_{name}.html")

sys.exit(0)

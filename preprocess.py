import pandas as pd
import argparse
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocess_data(data, label_column):
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in data.")
    
    # Separate features and label
    X = data.drop(columns=[label_column])
    y = data[label_column].reset_index(drop=True)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"[INFO] Numeric features: {len(numeric_features)}")
    print(f"[INFO] Categorical features: {len(categorical_features)} ({categorical_features})")

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names reliably from the preprocessor itself
    all_feature_names = preprocessor.get_feature_names_out()
    
    # Clean up names (remove 'num__' and 'cat__' prefixes for readability)
    all_feature_names = [name.split('__')[-1] for name in all_feature_names]
    
    preprocessed_df = pd.DataFrame(X_processed, columns=all_feature_names)


    # Concatenate label back
    final_df = pd.concat([preprocessed_df, y], axis=1)
    return final_df

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for ML.")
    parser.add_argument("--input", default="device1_top_20_features.csv", help="Input CSV file")
    parser.add_argument("--output", default="preprocessed_device1_features.csv", help="Output CSV file")
    parser.add_argument("--label", default="label", help="Name of the label column")
    args = parser.parse_args()

    try:
        print(f"[INFO] Loading data from {args.input}...")
        data = pd.read_csv(args.input)
        
        preprocessed_df = preprocess_data(data, args.label)
        
        preprocessed_df.to_csv(args.output, index=False)
        print(f"[SUCCESS] Preprocessed data saved to {args.output}")
        print(f"[INFO] Shape: {preprocessed_df.shape}")
    except FileNotFoundError:
        print(f"[ERROR] File {args.input} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


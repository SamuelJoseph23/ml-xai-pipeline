import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_data(csv_file):
    try:
        data = pd.read_csv(csv_file)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

def preprocess_data(data, label_column):
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in data.")
    
    # Separate features and label
    features = data.drop(columns=[label_column])
    label = data[label_column].reset_index(drop=True)
    
    numeric_features = features.select_dtypes(include='number').columns.tolist()
    # Optional: Warn or drop non-numeric columns
    non_numeric = list(set(features.columns) - set(numeric_features))
    if non_numeric:
        print(f"[INFO] Ignoring non-numeric features: {non_numeric}")

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)],
        remainder='drop'  # Ensures only numeric columns are kept
    )

    # Fit and transform the features
    preprocessed_data = preprocessor.fit_transform(features)
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features)

    # Concatenate label back to DataFrame
    final_df = pd.concat([preprocessed_df, label], axis=1)
    return final_df

def save_data(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save data: {e}")

def main():
    csv_file = 'device1_top_20_features.csv'
    output_file = 'preprocessed_device1_features.csv'
    label_column = 'label'  # Update to your actual label column name

    data = load_data(csv_file)
    if data is not None:
        try:
            preprocessed_df = preprocess_data(data, label_column)
            save_data(preprocessed_df, output_file)
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")

if __name__ == "__main__":
    main()

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
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, label_column):
    # Separate features and label
    features = data.drop(columns=[label_column])
    label = data[label_column]
    
    numeric_features = features.columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())                  
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)]
    )
 
    # Fit and transform the features
    preprocessed_data = preprocessor.fit_transform(features)

    # Create a DataFrame for the preprocessed features
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=numeric_features)
    
    # Concatenate the label back to the preprocessed features
    final_df = pd.concat([preprocessed_df, label.reset_index(drop=True)], axis=1)
    
    return final_df

def save_data(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    csv_file = 'device1_top_20_features.csv'
    output_file = 'preprocessed_device1_features.csv'
    label_column = 'label'  # Specify the name of your label column here

    data = load_data(csv_file)
    if data is not None:
        preprocessed_df = preprocess_data(data, label_column)
        save_data(preprocessed_df, output_file)

if __name__ == "__main__":
    main()

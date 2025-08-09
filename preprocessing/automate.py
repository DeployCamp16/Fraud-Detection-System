import pandas as pd
import numpy as np

from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline

def preprocess_data(save_path, file_path, pipeline_preprocessing_path, column_file_path):
    data = pd.read_csv(file_path)
    target_column = "label"

    data.drop(columns=["transaction_id"], inplace = True)
    data.drop_duplicates(inplace = True)

    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]
 
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create pipeline with RandomOverSampler
    pipeline = make_imb_pipeline(
        preprocessor,
        RandomOverSampler(random_state=42)
    )

    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    X_test = pipeline.named_steps['columntransformer'].transform(X_test)

    # Extract categorical feature names from OneHotEncoder
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_features = []
    for i, col in enumerate(categorical_features):
        cats = cat_encoder.categories_[i]
        cat_features.extend([f"{col}_{cat}" for cat in cats])

    dump(pipeline.named_steps['columntransformer'], pipeline_preprocessing_path)
    print(f'preprocessor pipeline successfully saved to: {pipeline_preprocessing_path}\n')

    # Save the cleaned data to a csv file
    preproc_data = pd.DataFrame(X_train, columns=numeric_features + cat_features)
    preproc_data['label'] = y_train.reset_index(drop=True)
    preproc_data.to_csv(save_path, index=False)

    # save preprocessed column names
    df_header = pd.DataFrame(columns=numeric_features + cat_features)
    df_header.to_csv(column_file_path, index=False)

    print(f"Column names successfully saved to : {column_file_path}")
    print(f'Pre-processed data successfully saved to: {save_path}\n')

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        file_path='../data/raw/fraud_detection.csv',
        save_path='../data/processed/fraud_detection_processed.csv',
        pipeline_preprocessing_path = '../models/preprocessing/preprocessor.joblib',
        column_file_path = '../data/processed/column_list_processed.csv'
    )
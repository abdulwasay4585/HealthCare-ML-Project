import pandas as pd
import numpy as np
from prefect import task, flow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
import os

# Constants
DATA_PATH = "data/diabetic_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@task(name="Load Data")
def load_data(filepath: str):
    print(f"Loading data from {filepath}...")
    # Replace '?' with NaN
    df = pd.read_csv(filepath, na_values=['?'])
    return df

@task(name="Preprocess Data")
def preprocess_data(df: pd.DataFrame):
    print("Preprocessing data...")
    
    # Drop columns with too many missing values or irrelevant
    drop_cols = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=['readmitted', 'time_in_hospital'])
    
    # Base Features
    # Feature Engineering/Selection
    all_numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                        'num_medications', 'number_outpatient', 'number_emergency', 
                        'number_inpatient', 'number_diagnoses']
    categorical_features = ['race', 'gender', 'age', 'admission_type_id', 
                            'discharge_disposition_id', 'admission_source_id', 
                            'diabetesMed', 'change']
    
    # Filter valid columns
    all_numeric_features = [f for f in all_numeric_features if f in df_clean.columns]
    categorical_features = [f for f in categorical_features if f in df_clean.columns]
    
    # Classification Features (Can use time_in_hospital)
    X_class = df_clean[all_numeric_features + categorical_features]
    
    # Regression Features (Must exclude time_in_hospital)
    reg_numeric_features = [f for f in all_numeric_features if f != 'time_in_hospital']
    X_reg = df_clean[reg_numeric_features + categorical_features]
    
    # Targets
    y_class = df_clean['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
    y_reg = df_clean['time_in_hospital']
    
    return X_class, X_reg, y_class, y_reg, all_numeric_features, reg_numeric_features, categorical_features

@task(name="Build Preprocessing Pipeline")
def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

@task(name="Train Classification Model")
def train_classification(X_train, y_train, preprocessor):
    print("Training Classification Model...")
    # Using RandomForest
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))])
    clf.fit(X_train, y_train)
    return clf

@task(name="Train Regression Model")
def train_regression(X_train, y_train, preprocessor):
    print("Training Regression Model...")
    reg = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', Ridge(alpha=1.0))])
    reg.fit(X_train, y_train)
    return reg

@task(name="Evaluate Models")
def evaluate_models(clf, reg, X_test_c, y_test_c, X_test_r, y_test_r):
    # Classification
    print("\n--- Classification Evaluation ---")
    y_pred_class = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_class)
    print(f"Classification Accuracy: {acc:.4f}")
    print(classification_report(y_test_c, y_pred_class))
    
    # Regression
    print("\n--- Regression Evaluation ---")
    y_pred_reg = reg.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))
    print(f"Regression RMSE: {rmse:.4f}")
    
    return {"accuracy": acc, "rmse": rmse}

@task(name="Save Models")
def save_models(clf, reg):
    joblib.dump(clf, os.path.join(MODEL_DIR, "classification_model.pkl"))
    joblib.dump(reg, os.path.join(MODEL_DIR, "regression_model.pkl"))
    print(f"Models saved to {MODEL_DIR}")

@flow(name="Healthcare Training Pipeline")
def training_flow():
    df = load_data(DATA_PATH)
    
    X_c, X_r, y_c, y_r, num_feats_c, num_feats_r, cat_feats = preprocess_data(df)
    
    # Split Classification
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    
    # Split Regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    
    # Build Pipelines
    preprocessor_c = build_pipeline(num_feats_c, cat_feats)
    preprocessor_r = build_pipeline(num_feats_r, cat_feats)
    
    # Train
    clf_model = train_classification(X_train_c, y_train_c, preprocessor_c)
    reg_model = train_regression(X_train_r, y_train_r, preprocessor_r)
    
    # Evaluate
    evaluate_models(clf_model, reg_model, X_test_c, y_test_c, X_test_r, y_test_r)
    
    # Save
    save_models(clf_model, reg_model)


if __name__ == "__main__":
    training_flow()

import pandas as pd
import numpy as np
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

def load_data(filepath: str):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, na_values=['?'])
    return df

def preprocess_data(df: pd.DataFrame):
    print("Preprocessing data...")
    drop_cols = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_clean = df_clean.dropna(subset=['readmitted', 'time_in_hospital'])
    
    all_numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                        'num_medications', 'number_outpatient', 'number_emergency', 
                        'number_inpatient', 'number_diagnoses']
    categorical_features = ['race', 'gender', 'age', 'admission_type_id', 
                            'discharge_disposition_id', 'admission_source_id', 
                            'diabetesMed', 'change']
    
    all_numeric_features = [f for f in all_numeric_features if f in df_clean.columns]
    categorical_features = [f for f in categorical_features if f in df_clean.columns]
    
    X_class = df_clean[all_numeric_features + categorical_features]
    reg_numeric_features = [f for f in all_numeric_features if f != 'time_in_hospital']
    X_reg = df_clean[reg_numeric_features + categorical_features]
    
    y_class = df_clean['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
    y_reg = df_clean['time_in_hospital']
    
    return X_class, X_reg, y_class, y_reg, all_numeric_features, reg_numeric_features, categorical_features

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

def train_classification(X_train, y_train, preprocessor):
    print("Training Classification Model...")
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))])
    clf.fit(X_train, y_train)
    return clf

def train_regression(X_train, y_train, preprocessor):
    print("Training Regression Model...")
    reg = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', Ridge(alpha=1.0))])
    reg.fit(X_train, y_train)
    return reg

def evaluate_models(clf, reg, X_test_c, y_test_c, X_test_r, y_test_r):
    print("\n--- Classification Evaluation ---")
    y_pred_class = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_class)
    print(f"Classification Accuracy: {acc:.4f}")
    print(classification_report(y_test_c, y_pred_class))
    
    print("\n--- Regression Evaluation ---")
    y_pred_reg = reg.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))
    print(f"Regression RMSE: {rmse:.4f}")
    
    return {"accuracy": acc, "rmse": rmse}

import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random
from datetime import datetime, timedelta

def train_clustering(X_train, preprocessor):
    print("Training Clustering Model (PCA + K-Means)...")
    # PCA to reducing to 2 components for simple visualization or just feature reduction
    # KMeans with 3 clusters (e.g., Low, Medium, High complexity patients?)
    cluster_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=2)),
        ('kmeans', KMeans(n_clusters=3, random_state=42))
    ])
    cluster_pipeline.fit(X_train)
    return cluster_pipeline

def prepare_time_series(df):
    print("Preparing Time Series Data...")
    # Simulate dates from 1999 to 2008
    start_date = datetime(1999, 1, 1)
    end_date = datetime(2008, 12, 31)
    days_range = (end_date - start_date).days
    
    # Generate random dates
    np.random.seed(42)
    random_days = np.random.randint(0, days_range, size=len(df))
    dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    df_ts = df.copy()
    df_ts['admission_date'] = dates
    df_ts = df_ts.sort_values('admission_date')
    
    # Aggregate admissions per month
    df_ts.set_index('admission_date', inplace=True)
    monthly_admissions = df_ts.resample('M').size()
    return monthly_admissions

def train_time_series(monthly_admissions):
    print("Training Time Series Model (ExponentialSmoothing)...")
    # Holt-Winters Exponential Smoothing
    model = ExponentialSmoothing(monthly_admissions, trend='add', seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()
    return fitted_model

def evaluate_clustering(model, X_test):
    # Clustering doesn't have "ground truth" labels usually in this context implies unsupervised
    preds = model.predict(X_test)
    print(f"Clustering predictions shape: {preds.shape}")
    return {} # Metric?

def evaluate_vis_forecast(model, monthly_admissions):
    # Forecast next 12 months
    forecast = model.forecast(12)
    print("Forecast for next 12 months:", forecast.values)
    return {"forecast": forecast.values.tolist()}

def save_models(clf, reg, cluster, ts_model, metrics):
    joblib.dump(clf, os.path.join(MODEL_DIR, "classification_model.pkl"))
    joblib.dump(reg, os.path.join(MODEL_DIR, "regression_model.pkl"))
    joblib.dump(cluster, os.path.join(MODEL_DIR, "cluster_model.pkl"))
    joblib.dump(ts_model, os.path.join(MODEL_DIR, "timeseries_model.pkl"))
    
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)
        
    print(f"Models and metrics saved to {MODEL_DIR}")

def main():
    df = load_data(DATA_PATH)
    X_c, X_r, y_c, y_r, num_feats_c, num_feats_r, cat_feats = preprocess_data(df)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    
    preprocessor_c = build_pipeline(num_feats_c, cat_feats)
    preprocessor_r = build_pipeline(num_feats_r, cat_feats)
    
    clf_model = train_classification(X_train_c, y_train_c, preprocessor_c)
    reg_model = train_regression(X_train_r, y_train_r, preprocessor_r)
    
    # Clustering (Use X_c features as general patient representation)
    cluster_model = train_clustering(X_train_c, preprocessor_c)
    
    # Time Series
    monthly_admissions = prepare_time_series(df)
    ts_model = train_time_series(monthly_admissions)
    
    metrics = evaluate_models(clf_model, reg_model, X_test_c, y_test_c, X_test_r, y_test_r)
    ts_metrics = evaluate_vis_forecast(ts_model, monthly_admissions)
    metrics.update(ts_metrics)
    
    save_models(clf_model, reg_model, cluster_model, ts_model, metrics)

if __name__ == "__main__":
    main()

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os
from contextlib import asynccontextmanager

# Paths
MODEL_DIR = "models"
CLF_MODEL_PATH = os.path.join(MODEL_DIR, "classification_model.pkl")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "regression_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

CLUSTER_MODEL_PATH = os.path.join(MODEL_DIR, "cluster_model.pkl")
TS_MODEL_PATH = os.path.join(MODEL_DIR, "timeseries_model.pkl")

# Global models
models = {}
templates = Jinja2Templates(directory="app/templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    try:
        if os.path.exists(CLF_MODEL_PATH):
            models["clf"] = joblib.load(CLF_MODEL_PATH)
            print("Classification model loaded.")
        if os.path.exists(REG_MODEL_PATH):
            models["reg"] = joblib.load(REG_MODEL_PATH)
            print("Regression model loaded.")
        if os.path.exists(CLUSTER_MODEL_PATH):
            models["cluster"] = joblib.load(CLUSTER_MODEL_PATH)
            print("Clustering model loaded.")
        if os.path.exists(TS_MODEL_PATH):
            models["ts"] = joblib.load(TS_MODEL_PATH)
            print("Time Series model loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")
    yield
    models.clear()

app = FastAPI(title="Healthcare ML API", version="1.0", lifespan=lifespan)

class PatientData(BaseModel):
    race: str = "Caucasian"
    gender: str = "Female"
    age: str = "[50-60)"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = 3
    num_lab_procedures: int = 40
    num_procedures: int = 0
    num_medications: int = 15
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0
    number_diagnoses: int = 5
    diabetesMed: str = "Yes"
    change: str = "Ch"

    class Config:
        extra = "ignore"

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/validation", response_class=HTMLResponse)
def read_validation(request: Request):
    return templates.TemplateResponse("validation_report.html", {"request": request})

@app.get("/metrics")
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {"accuracy": 0, "rmse": 0}

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/predict/readmission")
def predict_readmission(data: PatientData):
    if "clf" not in models:
        raise HTTPException(status_code=503, detail="Classification model not loaded")
    df = pd.DataFrame([data.dict()])
    try:
        prediction = models["clf"].predict(df)[0]
        result = "YES" if prediction == 1 else "NO"
        prob = models["clf"].predict_proba(df)[0].tolist()
        return {"prediction": result, "probabilities": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/los")
def predict_los(data: PatientData):
    if "reg" not in models:
        raise HTTPException(status_code=503, detail="Regression model not loaded")
    df = pd.DataFrame([data.dict()])
    try:
        prediction = models["reg"].predict(df)[0]
        return {"predicted_time_in_hospital": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/cluster")
def predict_cluster(data: PatientData):
    if "cluster" not in models:
        raise HTTPException(status_code=503, detail="Clustering model not loaded")
    df = pd.DataFrame([data.dict()])
    try:
        # Clustering uses same features as classification usually (or defined in pipeline)
        # Our training used X_train_c features. Pipeline handles it.
        cluster_id = models["cluster"].predict(df)[0]
        # Describe cluster?
        descriptions = {
            0: "Standard Case (Cluster 0)",
            1: "Complex/High-Resource (Cluster 1)",
            2: "Routine/Low-Resource (Cluster 2)"
        }
        return {"cluster_id": int(cluster_id), "description": descriptions.get(int(cluster_id), "Unknown")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/forecast")
def predict_forecast():
    if "ts" not in models:
        raise HTTPException(status_code=503, detail="Time Series model not loaded")
    try:
        # Forecast next 12 steps (months)
        forecast = models["ts"].forecast(12)
        # Convert to list
        return {"forecast": forecast.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

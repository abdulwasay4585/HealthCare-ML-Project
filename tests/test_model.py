import os
import joblib
import pytest

MODEL_DIR = "models"

@pytest.mark.skipif(not os.path.exists(MODEL_DIR), reason="Model directory not found")
def test_models_exist():
    clf_path = os.path.join(MODEL_DIR, "classification_model.pkl")
    reg_path = os.path.join(MODEL_DIR, "regression_model.pkl")
    
    # These verify the file existence
    assert os.path.exists(clf_path), "Classification model missing"
    assert os.path.exists(reg_path), "Regression model missing"

@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "classification_model.pkl")), reason="Models not trained")
def test_model_loading():
    clf = joblib.load(os.path.join(MODEL_DIR, "classification_model.pkl"))
    reg = joblib.load(os.path.join(MODEL_DIR, "regression_model.pkl"))
    
    assert clf is not None
    assert reg is not None

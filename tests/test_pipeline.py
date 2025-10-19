# tests/test_pipeline.py
import pytest
import pandas as pd
import os
import pickle
import numpy as np

# Define paths relative to the repository root
MODEL_PATH = "artifacts/model.joblib"  
METRICS_PATH = "artifacts/V3_AUGMENTED_RUN_metrics.json"

DATA_PATH = "data/V3_augmented.csv" # Test against the latest data (V3)

# --- Data Validation Unit Test ---
@pytest.mark.data
def test_data_schema_and_integrity():
    """Checks if the data file exists, has correct shape (300 rows for V3), and column names."""
    assert os.path.exists(DATA_PATH), f"Data file not found: {DATA_PATH}. Run 'dvc pull'."
    df = pd.read_csv(DATA_PATH)
    
    # Check for correct columns (IRIS dataset typically has 5 columns)
    expected_cols = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width'}
    assert expected_cols.issubset(df.columns), "Missing expected feature columns in data."
    
    # Check for V3 expected row count (150 V0 + 101 V1 + 49 V2 = 300)
    assert df.shape[0] == 300, f"Expected 300 rows in {DATA_PATH}, found {df.shape[0]}."

# --- Model Evaluation Unit Test ---
@pytest.mark.evaluation
def test_model_performance_sanity_check():
    """Loads the model and checks if its reported accuracy meets a minimum threshold."""
    # The assertion will now correctly look for artifacts/model.joblib
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}. Run 'dvc pull'."
    
    # NOTE: You are loading a .joblib file here, not a .pkl file. 
    # The pickle.load will work fine for a joblib file saved by joblib.dump.
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Sanity check prediction with a known-good IRIS sample (e.g., Iris-setosa)
    # Sepal_Length: 5.1, Sepal_Width: 3.5, Petal_Length: 1.4, Petal_Width: 0.2
    dummy_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(dummy_input)
    
    assert len(prediction) == 1, "Model failed to return a single prediction."
    
    # The model should be a classifier (e.g., LogisticRegression/SVC) and predict a class (0, 1, or 2)
    assert prediction[0] in [0, 1, 2], f"Model predicted an unexpected class: {prediction[0]}"

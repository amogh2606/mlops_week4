# tests/test_pipeline.py (Modified to use MLflow)

import pytest
import pandas as pd
import os
import mlflow.pyfunc # Needed for model loading
from mlflow.tracking import MlflowClient # Needed if we want to programmatically tag the best model
import numpy as np
from joblib import load

<<<<<<< HEAD
# NOTE: The model path is now an MLflow URI, not a file path.
MLFLOW_MODEL_URI = "models:/IRIS_DecisionTree_Model/Latest"
DATA_PATH = "data/V3_augmented.csv" 
# DVC pulls the data, but the model comes from MLflow
=======
# Define paths relative to the repository root
MODEL_PATH = "artifacts/model.joblib"  
METRICS_PATH = "artifacts/V3_AUGMENTED_RUN_metrics.json"

DATA_PATH = "data/V3_augmented.csv" # Test against the latest data (V3)
>>>>>>> c46026d8f130bc0e034f77f1346a03f6746e5241

# --- Data Validation Unit Test (Remains the same - uses local DVC pulled data) ---
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

# --- Model Evaluation Unit Test (Modified to use MLflow) ---
@pytest.mark.evaluation
def test_model_performance_sanity_check():
<<<<<<< HEAD
    """Loads the model from MLflow and runs sanity checks."""
    
    # 1. Load Model from MLflow Registry (Latest version)
    try:
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
        print(f"Successfully loaded model from MLflow: {MLFLOW_MODEL_URI}")
    except Exception as e:
        pytest.fail(f"Failed to load model from MLflow URI {MLFLOW_MODEL_URI}. Ensure MLflow server is running and model is registered. Error: {e}")

    # 2. SANITY PREDICTION
    dummy_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(dummy_input)

    # 3. ASSERTION
    assert len(prediction) == 1, "Model failed to return a single prediction."
    
    # Asserting the predicted label is a valid string class
    VALID_CLASSES = ['setosa', 'versicolor', 'virginica']
    
    assert prediction[0] in VALID_CLASSES, \
           f"Model predicted an unexpected class '{prediction[0]}'. Must be one of {VALID_CLASSES}"
    
    assert prediction[0] == 'setosa', \
           f"Model predicted {prediction[0]}, but expected 'setosa' for the dummy input."
=======
    """
    Loads the model artifact and performs a basic sanity check 
    to ensure it loads correctly and predicts a valid class label.
    """
    # 1. ASSERT FILE EXISTENCE (Success check from previous step)
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}. Run 'dvc pull'."

    # 2. LOAD MODEL USING JOBLIB (Fixes UnpicklingError)
    try:
        # Use joblib.load directly, as the model was saved using joblib.dump
        model = load(MODEL_PATH)
    except Exception as e:
        pytest.fail(f"Failed to load model from {MODEL_PATH} using joblib.load. Error: {e}")

    # 3. DEFINE SANITY INPUT
    # This input is a known sample for Iris-setosa: Sepal_Length: 5.1, Sepal_Width: 3.5, Petal_Length: 1.4, Petal_Width: 0.2
    dummy_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    
    # 4. GET PREDICTION
    prediction = model.predict(dummy_input)

    # 5. ASSERT PREDICTION SANITY (Fixes Assertion Error)
    assert len(prediction) == 1, "Model failed to return a single prediction."
    
    # Check if the predicted label is one of the expected string classes
    VALID_CLASSES = ['setosa', 'versicolor', 'virginica']
    
    # This assertion ensures the model output matches the format it was trained on (strings).
    assert prediction[0] in VALID_CLASSES, \
           f"Model predicted an unexpected class '{prediction[0]}'. Must be one of {VALID_CLASSES}"
    
    # Optional: A stronger test asserting the specific expected class for the sample input
    assert prediction[0] == 'setosa', \
           f"Model predicted {prediction[0]}, but expected 'setosa' for the dummy input."
>>>>>>> c46026d8f130bc0e034f77f1346a03f6746e5241

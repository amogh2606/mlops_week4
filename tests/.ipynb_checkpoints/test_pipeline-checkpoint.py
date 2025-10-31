# tests/test_pipeline.py (Modified to use MLflow)

import pytest
import pandas as pd
import os
import mlflow.pyfunc # Needed for model loading
from mlflow.tracking import MlflowClient # Needed if we want to programmatically tag the best model
import numpy as np

# NOTE: The model path is now an MLflow URI, not a file path.
MLFLOW_MODEL_URI = "models:/IRIS_DecisionTree_Model/Latest"
DATA_PATH = "data/V3_augmented.csv" 
# DVC pulls the data, but the model comes from MLflow

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
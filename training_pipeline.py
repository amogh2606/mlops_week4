import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
# Removed direct google.cloud.storage import, as DVC handles GCS interaction
from joblib import dump
import datetime
import os
import sys

# --- Configuration ---
# Local paths for DVC to track (artifacts will be saved here locally)
MODEL_OUTPUT_PATH = "artifacts/model.joblib"
LOGS_OUTPUT_PATH = "artifacts/logs.txt"

# --- Pipeline Core Function (Uses ONLY LOCAL paths) ---

def train_and_evaluate(local_data_path, model_output_path, logs_output_path, version_name):
    """
    Runs the full ML process, loads data from local_data_path, 
    trains the model, and saves artifacts locally.
    
    This function is agnostic to the data's remote location (GCS).
    """
    print(f"\n================ Starting Training on {version_name} ================")
    
    # --- Step 1: Load Data Locally (This assumes DVC has already pulled the file) ---
    try:
        data_df = pd.read_csv(local_data_path)
        print(f"Data loaded successfully from local path: {local_data_path}")
    except Exception as e:
        print(f"ERROR: Could not read local data file {local_data_path}. Ensure DVC pull was successful: {e}")
        return None, None, None
        
    # --- Step 2: Train and Evaluate Model ---
    try:
        # Assuming standard Iris column names and structure
        train, test = train_test_split(data_df, test_size=0.4, stratify=data_df['species'], random_state=42)
    except KeyError:
        print("ERROR: 'species' column not found in DataFrame. Check data file structure.")
        return None, None, None

    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species

    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    prediction = mod_dt.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    
    print(f'Model trained successfully. Accuracy: {accuracy:.3f}')
    
    # --- Step 3: Save Artifacts Locally (DVC tracks this output) ---
    
    # Ensure artifacts directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Save model
    dump(mod_dt, model_output_path)
    
    # Save logs/metrics
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_content = (
        f"Training version: {version_name}\n"
        f"Execution time: {timestamp}\n"
        f"Model accuracy: {accuracy:.3f}\n"
        f"Data Source: {local_data_path}"
    )
    with open(logs_output_path, "w") as f:
        f.write(log_content)
    
    print(f"\nArtifacts saved locally to '{os.path.dirname(model_output_path)}/'")

    return mod_dt, accuracy, timestamp

# --- DVC-Compliant Execution (Runs one job based on command line input) ---

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python training_pipeline.py <local_data_path> <version_name>")
        print("Example: python training_pipeline.py data/V0_iris.csv V0_RUN1")
        sys.exit(1)
        
    local_data_path = sys.argv[1]
    version_name = sys.argv[2]
    
    model, accuracy, timestamp = train_and_evaluate(
        local_data_path=local_data_path,
        model_output_path=MODEL_OUTPUT_PATH,
        logs_output_path=LOGS_OUTPUT_PATH,
        version_name=version_name
    )

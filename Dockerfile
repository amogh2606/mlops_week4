# --- Stage 1: Production (Lean, runtime image) ---
FROM python:3.11-slim

# 1. Set the working directory for the final application
WORKDIR /app

# 2. Install necessary system packages and Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
# Install production Python dependencies (FastAPI, joblib, scikit-learn, etc.)
RUN pip install --no-cache-dir -r requirements.txt gunicorn uvicorn

# 3. Copy application code
COPY app.py .

# 4. Copy the model artifact from the Git repository
# Source: artifacts/model.joblib (now tracked in Git)
# Destination: /app/artifacts/model.joblib (for app.py to load)
COPY artifacts/model.joblib artifacts/model.joblib

# Set port and start command
ENV PORT 8080
EXPOSE 8080

# Use Gunicorn with Uvicorn workers for robust production deployment
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker"]
FROM python:3.12-slim

WORKDIR /app

# Ensure Python outputs logs immediately to Render console
ENV PYTHONUNBUFFERED=1

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and UI templates
COPY src/ ./src/

# Trained model artifacts & decision boundary
COPY model.pkl preprocessor.pkl metrics.json .decision_threshold ./

# Dataset profile, split info, and stationarity reports
COPY dataset_profile.json split_info.json stationarity_report.json stationarity_report.csv ./

# Subgroup metrics for dashboard
COPY subgroup_metrics.csv ./

# MLflow tracking store so the dashboard has complete run history
COPY mlruns/ ./mlruns/

# params.yaml for monitoring thresholds
COPY params.yaml .

# predictions.jsonl is created at runtime for streaming logs
RUN touch predictions.jsonl

# Dynamic port binding for Render ($PORT) with fallback to 8000 for local execution
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

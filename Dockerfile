FROM python:3.12-slim

WORKDIR /app

# Ensure Python outputs logs immediately and unbuffered
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install essential system dependencies (libgomp1 is strictly required by XGBoost on Debian slim; curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and observatory module
COPY src/ ./src/

# Copy helper scripts (seed_demo, etc.)
COPY scripts/ ./scripts/

# Copy trained model artifacts and calibrated decision boundary
COPY model.pkl preprocessor.pkl metrics.json .decision_threshold ./

# Copy dataset profile, split info, and stationarity reports
COPY dataset_profile.json split_info.json stationarity_report.json stationarity_report.csv ./

# Copy subgroup metrics and evaluation artifacts
COPY subgroup_metrics.csv ./
COPY results/ ./results/

# Copy MLflow tracking store
COPY mlruns/ ./mlruns/

# Copy configuration files
COPY params.yaml observatory.yaml ./

# Prepare runtime write directories and empty predictions log
RUN mkdir -p /app/runs /app/results /app/secrets && touch /app/predictions.jsonl

# Expose local default (8000) and Render default (10000)
EXPOSE 8000
EXPOSE 10000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

# Launch FastAPI app with dynamic port binding for Render ($PORT) and local (8000)
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

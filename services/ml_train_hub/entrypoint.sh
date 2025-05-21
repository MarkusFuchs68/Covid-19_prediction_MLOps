#!/bin/bash

# Set default stage if not provided
RUNNING_STAGE="${RUNNING_STAGE:-dev}"

# Define ports based on environment
if [ "$RUNNING_STAGE" = "prod" ]; then
  MLFLOW_PORT=8081
  UVICORN_PORT=8082
else
  MLFLOW_PORT=8001
  UVICORN_PORT=8002
fi

echo "🚀 Starting in '$RUNNING_STAGE' mode..."
echo "MLflow will run on port $MLFLOW_PORT"
echo "FastAPI will run on port $UVICORN_PORT"

# Start MLflow in the background
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 0.0.0.0 --port $MLFLOW_PORT &

# Start FastAPI in the foreground
uvicorn ml_train_hub.app.main:app --host 0.0.0.0 --port $UVICORN_PORT

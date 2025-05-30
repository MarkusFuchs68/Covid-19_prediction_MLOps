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

# Start MLflow server in background
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 0.0.0.0 --port $MLFLOW_PORT &

# Start FastAPI server in background
uvicorn ml_train_hub.app.main:app --host 0.0.0.0 --port $UVICORN_PORT &

# Wait for FastAPI (Uvicorn) server
echo "Waiting for Uvicorn server to be ready..."
until curl -s "http://localhost:$UVICORN_PORT/docs" > /dev/null; do
  echo "Uvicorn not ready yet..."
  sleep 5
done
echo "✅ Uvicorn is ready!"

# Wait for MLflow server
echo "Waiting for MLflow server to be ready..."
until curl -s "http://localhost:$MLFLOW_PORT" > /dev/null; do
  echo "MLflow not ready yet..."
  sleep 5
done
echo "✅ MLflow is ready!"

# Run all tests including the integration tests and output all logger messages except DEBUG level
pytest -o log_cli=true -o log_cli_level=INFO

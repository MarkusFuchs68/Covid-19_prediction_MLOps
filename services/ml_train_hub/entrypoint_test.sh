#!/bin/bash

# start the mlflow server (in the background)
mlflow server --host 0.0.0.0 --port 8081 &

# start the FastAPI server (in the foreground, so the container stays running)
uvicorn main:app --host 0.0.0.0 --port 8082 &

# Wait for uvicorn (FastAPI) server
echo "Waiting for Uvicorn server to be ready..."
until curl -s http://localhost:8082/docs > /dev/null; do
  echo "Uvicorn not ready yet..."
  sleep 1
done
echo "✅ Uvicorn is ready!"

# Wait for MLflow server
echo "Waiting for MLflow server to be ready..."
until curl -s http://localhost:8081 > /dev/null; do
  echo "MLflow not ready yet..."
  sleep 1
done
echo "✅ MLflow is ready!"

# Run all tests
pytest

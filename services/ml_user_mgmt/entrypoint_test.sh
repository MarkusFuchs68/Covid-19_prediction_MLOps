#!/bin/bash

# Set default stage if not provided
RUNNING_STAGE="${RUNNING_STAGE:-dev}"

# Define ports based on environment
if [ "$RUNNING_STAGE" = "prod" ]; then
  UVICORN_PORT=8083
else
  UVICORN_PORT=8003
fi

echo "🚀 Starting in '$RUNNING_STAGE' mode..."
echo "FastAPI will run on port $UVICORN_PORT"

# Start FastAPI server in background
uvicorn ml_user_mgmt.app.main:app --host 0.0.0.0 --port $UVICORN_PORT &

# Wait for FastAPI (Uvicorn) server
echo "Waiting for Uvicorn server to be ready..."
until curl -s "http://localhost:$UVICORN_PORT/docs" > /dev/null; do
  echo "Uvicorn not ready yet..."
  sleep 5
done
echo "✅ Uvicorn is ready!"

# Run all tests including the integration tests and output all logger messages except DEBUG level
pytest -o log_cli=true -o log_cli_level=INFO

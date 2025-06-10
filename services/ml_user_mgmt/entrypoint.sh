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

# Start FastAPI in the foreground
uvicorn ml_user_mgmt.app.main:app --host 0.0.0.0 --port $UVICORN_PORT

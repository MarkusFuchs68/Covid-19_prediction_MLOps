#!/bin/bash

# start the mlflow server (in the background)
mlflow server --host 0.0.0.0 --port 8081 &

# start the FastAPI server (in the foreground, so the container stays running)
uvicorn main:app --host 0.0.0.0 --port 8082

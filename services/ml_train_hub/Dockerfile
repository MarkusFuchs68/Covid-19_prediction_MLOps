### create those images with:
# dev: docker image build --target dev -t ml_train_hub:dev .
# test: docker image build --target test -t ml_train_hub:test .
# prod: docker image build --target prod -t ml_train_hub:<version_number> .
### run them with (-d = detached) from the ml_train_hub folder!:
# dev: docker run -d --rm -p 8001-8002:8001-8002 -v ./mlruns:/home/services/ml_train_hub/mlruns --name ml_train_hub_dev ml_train_hub:dev
# test: docker run --rm -p 8001-8002:8001-8002 --name ml_train_hub_test ml_train_hub:test
# prod: docker run -d --rm -p 8081-8082:8081-8082 -v ./mlruns:/home/services/ml_train_hub/mlruns --name ml_train_hub_prod ml_train_hub:<version_number> ./entrypoint.sh

# ----- BASE BUILD STAGE -----
    FROM python:3.11.9-slim AS builder

    # set work directory
    WORKDIR /home/services/ml_train_hub
    # This sets the project root, so import paths do work
    ENV PYTHONPATH=/home/services

    # set up environment variables
    ENV PYTHONUNBUFFERED=1
    ENV PYTHONDONTWRITEBYTECODE=1

    RUN apt-get update \
        # dependencies for building Python packages
        && apt-get install -y build-essential

    # copy project packages and requirements
    COPY ./requirements.txt .
    # install packages and requirements
    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt

    # copy the services files
    COPY ./app ./app

    # ----- DEV STAGE -----
    # this provides a running API and MLFlow server for local development
    FROM builder AS dev
    # setup env vars
    ENV LOG=1
    ENV RUNNING_STAGE=dev
    # copy startup script
    COPY __init__.py .
    COPY ./entrypoint.sh .
    CMD ["./entrypoint.sh"]

    # # ----- TEST STAGE -----
    # Stages used for testing later in CI/CD pipeline
    FROM builder AS test
    # setup env vars
    ENV LOG=1
    ENV RUNNING_STAGE=test
    # copy and install test requirements
    COPY ./requirements_test.txt .
    RUN pip install -r requirements_test.txt
    # copy test folder
    COPY ./tests ./tests
    # Add curl for the integration tests, it's used by the entrypoint script
    RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
    # copy startup script
    COPY __init__.py .
    COPY ./entrypoint_test.sh .
    CMD ["./entrypoint_test.sh"]

    # # ----- PROD STAGE -----
    FROM builder AS prod
    # usually you would copy app files, but for here we just copy test files
    # setup env vars
    ENV LOG=0
    ENV RUNNING_STAGE=prod
    # create a non-root user and switch to it
    RUN useradd produser
    USER produser
    # copy startup script
    COPY __init__.py .
    COPY ./entrypoint.sh .

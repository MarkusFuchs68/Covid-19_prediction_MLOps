# ----- BASE BUILD STAGE -----
FROM python:3.11.9-slim AS builder

# set work directory
WORKDIR /home/services

# set up environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    # dependencies for building Python packages
    && apt-get install -y build-essential

# copy only requirements.txt first to leverage caching
COPY ./requirements.txt .

# install packages and requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy the rest of the application files
COPY __init__.py ./ml_train_hub
COPY ./app ./ml_train_hub/app

# copy the evaluation dataset
COPY ./data ./ml_train_hub/data


# ----- DEV STAGE -----
FROM builder AS dev
# setup env vars
ENV LOG=1
ENV RUNNING_STAGE=dev

# Startup-script has to be run from parent folder copy to parent folder
COPY ./entrypoint.sh .
CMD ["./entrypoint.sh"]


# ----- TEST STAGE -----
FROM builder AS test
# setup env vars
ENV LOG=1
ENV RUNNING_STAGE=test

# copy and install test requirements
COPY ./requirements_test.txt .
RUN pip install -r requirements_test.txt

# copy test folder
COPY ./tests ./ml_train_hub/tests

# Add curl for the integration tests, it's used by the entrypoint script
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# copy startup script
COPY ./entrypoint_test.sh .
CMD ["./entrypoint_test.sh"]


# ----- PROD STAGE -----
FROM builder AS prod
# setup env vars
ENV LOG=0
ENV RUNNING_STAGE=prod

# create a non-root user and switch to it
RUN useradd produser
USER produser

# copy startup script
COPY ./entrypoint.sh .

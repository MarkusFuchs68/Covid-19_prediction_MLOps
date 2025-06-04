### create those images with:
# dev: docker image build --target dev -t ml_auth:dev .
# test: docker image build --target test -t ml_auth:test .
# prod: docker image build --target prod -t ml_auth:<version_number> .
### run them with (-d = detached) from the ml_auth folder!:
# dev: docker run -d --rm -p 8003:8003 --name ml_auth_dev ml_auth:dev
# test: docker run --rm -p 8003:8003 --name ml_auth_test ml_auth:test
# prod: docker run -d --rm -p 8083:8083 --name ml_auth_prod ml_auth:<version_number>

# ----- BASE BUILD STAGE -----
FROM python:3.11.9-slim AS builder

# set work directory
WORKDIR /home/services/ml_auth
ENV PYTHONPATH=/home/services

# set up environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# to surpress annoying cache size message, when using colima docker daemon
ENV OPENBLAS_VERBOSE=0

# dependencies for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# copy only requirements.txt first to leverage caching
COPY ./requirements.txt .

# install packages and requirements
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy the rest of the application files
COPY __init__.py .
COPY ./app ./app


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

# Add curl for the integration tests, it's used by the entrypoint script
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# copy and install test requirements
COPY ./requirements_test.txt .
RUN pip install -r requirements_test.txt

# copy test folder
COPY ./tests ./tests

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

# copy startup script, but don't run it automatically
# this is done in the docker-compose file
COPY ./entrypoint.sh .

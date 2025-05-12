# ----- BASE BUILD STAGE -----
FROM python:3.11.9-slim AS builder

# set work directory
WORKDIR /usr/src/app

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

# ----- DEV STAGE -----
FROM builder AS dev
ENV LOG=1
# copy service files
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# # ----- TEST STAGE -----
# FROM builder AS test
# COPY ./tests/ .
# # setup env vars
# ENV LOG=1

# # ----- PROD STAGE -----
FROM builder AS prod
# usually you would copy app files, but for here we just copy test files
COPY main.py .
# setup env vars
ENV LOG=0
# create a non-root user and switch to it
RUN useradd produser
USER produser

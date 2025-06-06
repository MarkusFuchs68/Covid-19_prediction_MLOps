# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the Covid-19 prediction MLOps project. These workflows automate the CI/CD pipeline based on the Git flow defined in the project.

## Branch Naming Convention

As defined in the project README, the branch naming convention is:

```
<stage>-<name>-<feature>
```

Example: `dev-vishal-ci-cd`

## Available Workflows

### 1. ML Host Backend CI/CD Pipeline (`ml_host_backend_pipeline.yml`)

A complete CI/CD pipeline for the ML Host Backend service that builds, tests, and deploys the service based on the branch name.

**Triggers:**
- Push to `main` or `dev-*-*` branches that changes files in `services/ml_host_backend/`
- Pull requests to `main` or `dev-*-*` branches that changes files in `services/ml_host_backend/`

**Stages:**
- For `main` branch: Builds and deploys production image
- For `dev-*-*` branches: Builds and deploys development image
- For other branches: Runs tests only

### 2. ML Train Hub CI/CD Pipeline (`ml_train_hub_pipeline.yml`)

A complete CI/CD pipeline for the ML Train Hub service that builds, tests, and deploys the service based on the branch name.

**Triggers:**
- Push to `main` or `dev-*-*` branches that changes files in `services/ml_train_hub/`
- Pull requests to `main` or `dev-*-*` branches that changes files in `services/ml_train_hub/`

**Stages:**
- For `main` branch: Builds and deploys production image, pulls DVC data
- For `dev-*-*` branches: Builds and deploys development image, pulls DVC data
- For other branches: Runs tests only

### 3. ML User Management CI/CD Pipeline (`ml_user_mgmt_pipeline.yml`)

A complete CI/CD pipeline for the ML User Management service that builds, tests, and deploys the service based on the branch name.

**Triggers:**
- Push to `main` or `dev-*-*` branches that changes files in `services/ml_user_mgmt/`
- Pull requests to `main` or `dev-*-*` branches that changes files in `services/ml_user_mgmt/`

**Stages:**
- For `main` branch: Builds and deploys production image
- For `dev-*-*` branches: Builds and deploys development image
- For other branches: Runs tests only

## How to Use

### Automatic Triggers

The workflows are automatically triggered based on the branch name and the files that are changed. For example, if you push changes to the `services/ml_host_backend/` directory on a branch named `dev-john-feature`, the `ml_host_backend_pipeline.yml` workflow will be triggered and will build and deploy a development image.



### Required Secrets

The following secrets need to be configured in the GitHub repository:

- `DVC_ACCESS_KEY_ID` and `DVC_SECRET_ACCESS_KEY`: For DVC operations with DAGsHub
  - These can be obtained from the DAGsHub repository (MarkusFuchs68/Covid-19_prediction_MLOps) under Remote → Data → DVC
- Docker registry credentials: For pushing images to a registry (if implementing image publishing)
- Deployment credentials: For deploying to your environments (if implementing actual deployment)

## Customization

These workflows are designed to be a starting point for your CI/CD pipeline. You may need to customize them based on your specific requirements, such as:

- Adding additional tests
- Configuring deployment to specific environments
- Setting up notifications
- Adding code quality checks

# Covid-19_prediction_MLOps

In this repository we collaborate on the Covid-19 detection project a with focus on MLOps

# Development Best Practices

## Python
We're using Python version 3.11.9 for now.
In later stage it is possible to select different python versions for each service (if required).

- For now you can create venv in root of the repository. In case we have need different venv for each service in local setup aswell it is possible to init venv at service level.
- Install requirements_development.txt for local setup
<code>
pip install -r requirements_development.txt
</code>

## Git Flow

- Branch Naming Convention: <stage>-<name>-<feature>
- E.g.: dev-phil-proj_structure

### Pre-Commit Hooks
Pre-Commit Hooks allow us to automatize best practices and coding standards for the entire team.

#### Setup
- In terminal with active venv type: <code>pre-commit install</code>

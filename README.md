# Covid-19_prediction_MLOps

In this repository we collaborate on the Covid-19 detection project a with focus on MLOps

# Development Best Practices

## Folder Structure
- services: Place for all containerized services for later deployments
- prototyping: Place for anything (notebooks, scripts, whatever) does not require testing and coding standards can be ignored.
- .envs: Place for local env files (e.g. later used for shared docker-compose setup) We can control with gitignore wether files should be on git or not, still: Be cautious using security related files!
- .dvc: Related to setup for data version control.

## Python
We're using Python version 3.11.9 for now.
In later stage it is possible to select different python versions for each service (if required).

- For now you can create venv in root of the repository. In case we have need different venv for each service in local setup aswell it is possible to init venv at service level. <code>python -m venv venv</code>
- Install requirements_dev.txt for local setup
<code>
pip install -r requirements_dev.txt
</code>

## Git Flow

- Branch Naming Convention: <code>\<stage>-\<name>-\<feature></code> e.g.: dev-phil-proj_structure

- check status: <code>git status</code>
- New branch: <code>git checkout -b \<branch_name\></code> after that: <code>git push --set-upstream origin \<branch_name\></code>
- Switch branch <code>git checkout \<branch_name\></code>
- Pull changes: <code>git pull</code> (alternative: fetch)
- Add changes: <code>git add --all</code> or <code>git add \<file\></code>
- Commit changes: <code>git commit -m \<message\></code>
- Push changes: <code>git push</code>
- Merge branch: <code>git merge \<branch_name\></code>


### Pre-Commit Hooks
Pre-Commit Hooks allow us to automatize (and force) best practices, coding standards, etc. for the entire team. All configurations are centralized in this file: <code>.pre-commit-config.yaml</code>.

Pre-Commit Hooks detect changes and will execute configured pipelines on changed files. If you want to run pre-commit hooks on the entire directory run: <code>pre-commit run --all</code> in your active python environment (terminal)

#### Setup
- In terminal with active venv type: <code>pre-commit install</code>
- Note: Make sure you have requirements_development.txt installed, in some cases this will be required (e.g. if we build our own local packages in future)

#### Behaviour
- Pipelines will be triggered automatically in in most cases errors will be fixed automatically through python packages(sorting, formatting, ...).
- If pipeline "failed" and changes automatically occured, you need to add new changes and than commit again which will retrigger the pre-commit pipelines. It is normal to repeat this process multiple times.
- Some packages, such as flake8 will not automatically fix errors, in that case you might need to fix the error manually. E.g. F401 imported but unused requires you to delete unused import yourself.
- <code>setup.cfg</code> is the configuration file to control behaviour for certain hooks.

## Docker-Compose
- <code>docker-compose up -d --build</code>
  * -d to run in detached mode
  * --build to make sure it repeats building file after code changes
- Execute pytest on container<code>docker-compose exec ml_host_backend_dev pytest .</code>
- Connect to container <code>docker compose exec -it ml_host_backend_dev sh</code>
- Run single services:
  * <code>docker-compose up -d ml_train_hub_\<choice: dev, test, prod, <version_number>></code>
  * <code>docker-compose up -d ml_host_backend_\<choice: dev, test, prod, <version_number>></code>

## Docker
- In most cases we build, tag and run containers through docker-compose
- List containers <code>docker ps</code>
- Connect to container <code>docker exec -it \<id\> sh</code> e.g. <code>docker exec -it c223c30b3fb8 sh</code>

## ml_train_hub service
- for local mlflow server, run this command from the **services/ml_train_hub** folder: <code>mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 0.0.0.0 --port 8001</code>
- for local mlflow server via docker: <code>docker-compose up ml_train_hub_dev</code>
- testing:
  - unit tests: <code>pytest</code>
  - integration tests: <code>pytest -m integration</code>

## DVC
for our projects demo purpose, we use DVC to pragmatically store the MLFlow database files in dagshub
initialize DVC:
- <code>pip install dvc</code>
- <code>pip install 'dvc[s3]'</code>
- for access key setup, go to: https://dagshub.com/MarkusFuchs68/Covid-19_prediction_MLOps, and click on **Remote/Data/DVC** and copy the code for 'Setup credentials' and run it.
- <code>dvc pull<br>dvc commit<br>dvc push</code>

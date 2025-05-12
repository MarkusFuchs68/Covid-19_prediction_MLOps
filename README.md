# Covid-19_prediction_MLOps

In this repository we collaborate on the Covid-19 detection project a with focus on MLOps

# Development Best Practices

## Python
We're using Python version 3.11.9 for now.
In later stage it is possible to select different python versions for each service (if required).

- For now you can create venv in root of the repository. In case we have need different venv for each service in local setup aswell it is possible to init venv at service level. <code>python -m venv venv</code>
- Install requirements_development.txt for local setup
<code>
pip install -r requirements_development.txt
</code>

## Git Flow

- Branch Naming Convention: <stage>-<name>-<feature>
- E.g.: dev-phil-proj_structure

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

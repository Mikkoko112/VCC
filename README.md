# VCC
Repository for the University of Hamburg team participating in the Virtual Cell Challenge.

## Initial Setup
quick tutorial on setting up a local work enviroment for this project

### git
1. make new local directory at desired location and navigate to it
2. make local copy of this remote repo with 'git clone "https://github.com/Mikkoko112/VCC.git"
3. navigate to /VCC with 'cd VCC' (local directory and the local copy shouldn't have the same name, e.g. .VCC/VCC/ is bad)

### venv and python package
4. create new virtual enviroment with 'python -m venv .venv'
5. activate said venv with 'source .venv/bin/activate'
6. update all installing tools with 'pip install --upgrade pip setuptools wheel'
7. install all dependencies and the codebase in src/ as a local python package with "pip install -e ."

### optional - Jupyter Notebooks
8. 'pip install ipykernel'
9. 'python -m ipykernel install --user --name vcc-uhh --display-name "Python (vcc-uhh)"'

Now you can import classes and functions across files and directories in src/ with 'from vcc-uhh import ...'
Additionally for this to work, every directory in src/ needs an '__init__.py' file (this can be empty)

## Maintenance
### Adding dependencies
Whenever you use a new package, add it to the pyproject.toml file and reinstall the local package like in step 7 of the setup
(Since the -e flag makes the package editable, you don't need to reinstall the package every time you change the codebase)

### Raw Data Handling
the data folder is ignored by the git repo, since raw data shouldn't be up for change. 
Instead, all data should be uploaded to and downloaded from the sharecloud and manually added to the data folder.

### Model freezing
If a model is a sufficient checkpoint, you can save the exact dependecies by using 'pip freeze > requirements.txt'

## License
MIT
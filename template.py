import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] : [%(message)s]")

project_name = "ecan"
project_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/resources/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dockerfile",
    "params.yaml",
    "dvc.yaml",
    "requirements.txt",
    "template/index.py",
    "setup.py",
    "researches/trials.ipynb",
]

for file_path in project_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory {file_dir} for the file {file_name}")

    if (not os.path.exists(file_path) or (os.path.getsize(file_path) == 0)):
        with open(file_path, "w"):
            pass
        logging.info(f"Creating empty file {file_path}")
    else:
        logging.info(f"The file {file_name} already exist !")

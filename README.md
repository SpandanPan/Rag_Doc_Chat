# Assignment 

# Added functionalities

```bash
# Added functionality to ingest csv,xlsx,pptx
Refer to src/document_ingestion/data_ingestion.py

# Added functionality to ingest images/table
Refer to src/document_ingestion/data_ingestion.py
Check at additional_files/check_image_table_handler.py

# Added Unit Tests 
Refer to /tests/test_comprehensive.py

# Added langchain chache
Refer to main/src/document_chat/retrieval.py

# Added pre-commit/post-commit hooks
Refer to .pre-commit-config.yaml
Added post commit hooks as well  - can test on commit
# Added additional tests apart from ci.yaml on push/PR
Refer to .github/workflows/tests.yml  - For checks on push to main

# Added Login screen
Refer to index.html
username - admin
password - 1234

# Added evaluation metric using deepeval 
Only included correctness, can add other metrics like relevance etc. as I ran out of token limit, as deepeval requires openai api key
Refer to additioanl_files/deep_eval_rag.py

# Project Setup Guide

## Create Project Folder and Environment Setup

```bash
# Create a new project folder
mkdir <project_folder_name>

# Move into the project folder
cd <project_folder_name>

# Open the folder in VS Code
code .

# Create a new Conda environment with Python 3.10
conda create -p <env_name> python=3.10 -y

# Activate the environment (use full path to the environment)
conda activate <path_of_the_env>

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Initialize Git
git init




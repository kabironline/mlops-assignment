week7/
├──poisoned_data_experiment.sh - A bash script that runs the poisoning experiments at different levels and settings
├──main.py - Entry point for the FastAPI Iris prediction service
├──Dockerfile - Docker configuration for containerizing the FastAPI service
├──k8s
│   └── deployment.yaml - Kubernetes deployment configuration for the FastAPI service
│   └── service.yaml - Kubernetes service deployment configuration for the FastAPI service
│   └── hpa.yaml - Kubernetes HorizontalPodAutoscaler configuration for the FastAPI service
├──.gitignore - specifies files to be ignored by git
├──.github
│   └── workflows
│       └── ci-dev.yml - GitHub Actions workflow for CI/CD executed when pushed to dev branch
│       └── ci-main.yml - GitHub Actions workflow for CI/CD executed when pushed to main branch
│       └── ci-week6.yml - Workflow for CI/CD executed when pushed to week6 branch, For automated building of Docker image and pushing to Docker Hub and then deploying to Kubernetes cluster in GKE
│       └── ci-week7.yml - Workflow for CI/CD executed when pushed to week7 branch, For automated building of Docker image and pushing to Docker Hub and then deploying to Kubernetes cluster in GKE and wrk load testing
│       └── cml-report.yml - Workflow 
├── data
│   └── iris.csv - data file for the Iris dataset
├── requirements.txt - list of dependencies for the project
├── src
│   ├── evaluate.py - script for model evaluation
│   ├── __init__.py
│   └── train.py - script for model training
│   └── train_poisoned_data.py - script for training poisoning data and training model
├── wrk_script.lua - wrk example load script
└── tests
    ├── __init__.py
    ├── test_data_validation.py - tests for data validation
    └── test_evaluation.py - tests for model evaluation

# Titanic MLOps Pipeline

This project implements an end-to-end MLOps pipeline using Spark, DVC, MLflow, and FastAPI to classify Titanic passengers based on survival.

## Features
- Spark-based preprocessing and training
- MLflow experiment tracking + model registry
- Drift detection + retraining support
- REST API for inference
- Data & model versioning with DVC
- Dockerized API deployment

## Run Locally

```bash
# 1. Preprocess, Train, and Simulate Drift
python main.py

# 2. Run FastAPI server
python api_server.py
---

## Docker Deployment

docker build -t titanic-api .
docker run -p 8000:8000 titanic-api

## Data Versioning

dvc add data/raw/titanic.csv
dvc add models/logistic_model/
dvc push


#!/bin/bash
set -e

echo "Setting up project with DVC..."

git init
git remote add origin https://your-git-repo-url 2>/dev/null || true

dvc pull --force
echo "DVC data pulled successfully."

echo "Running preprocessing & training pipeline..."
dvc repro

echo "Waiting for MLflow to be ready..."
sleep 10

echo "Starting FastAPI server..."
uvicorn api_server:app --host 0.0.0.0 --port 8000

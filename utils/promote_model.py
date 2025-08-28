# utils/promote_model.py

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "TitanicSparkModel"
STAGE = "Staging"

# Set up the MLflow client
client = MlflowClient()

# Get latest version of the registered model
latest_versions = client.get_latest_versions(name=MODEL_NAME, stages=["None"])

if not latest_versions:
    print(f"No unassigned versions found for model '{MODEL_NAME}'.")
else:
    version_to_promote = latest_versions[0].version
    print(f"Promoting version {version_to_promote} to stage '{STAGE}'...")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version_to_promote,
        stage=STAGE,
        archive_existing_versions=True  # optional: auto-archive previous Staging/Production models
    )

    print(f"Model '{MODEL_NAME}' version {version_to_promote} promoted to {STAGE}.")

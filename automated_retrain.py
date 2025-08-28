# automated_retrain.py

from pyspark.sql import SparkSession
import os
from src.train_model import train_model
from src.drift_detection import run_drift_detection
from src.data_preprocessing import preprocess_data

DRIFTED_DATA_PATH = "data/processed/titanic_preprocessed_drifted"
BASELINE_STATS_PATH = "data/baseline_stats"
MODEL_PATH = "models/latest_model"
THRESHOLD = 0.1

def automated_retrain():
    spark = SparkSession.builder.appName("AutomatedRetraining").getOrCreate()

    print("Running drift detection...")
    drift_report = run_drift_detection(spark, DRIFTED_DATA_PATH, BASELINE_STATS_PATH, threshold=THRESHOLD)

    drift_detected = any(info["drift"] for info in drift_report.values())
    if drift_detected:
        print("Drift detected! Retraining model...")

        # Retrain on new (drifted) data
        train_model(spark, DRIFTED_DATA_PATH, MODEL_PATH)
        print("Model retrained and logged via MLflow.")
    else:
        print("No significant drift detected. No retraining necessary.")

    spark.stop()

if __name__ == "__main__":
    automated_retrain()

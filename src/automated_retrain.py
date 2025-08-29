# src/automated_retrain.py

from pyspark.sql import SparkSession
import logging
import sys
import yaml
import os

from src.train_model import train_model
from src.drift_detection import run_drift_detection
from src.data_preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def automated_retrain(spark, DRIFTED_DATA_PATH, PROCESSED_PATH, BASELINE_STATS_PATH, MODEL_PATH, DRIFT_OUT, THRESHOLD=0.1):

    try:
            
        preprocess_data(spark, DRIFTED_DATA_PATH, PROCESSED_PATH)

        print("Pre-processing complete for the latest data.")

        print("Running drift detection...")
        # drift_report = run_drift_detection(spark, DRIFTED_DATA_PATH, BASELINE_STATS_PATH, threshold=THRESHOLD)
        drift_report = run_drift_detection(spark, PROCESSED_PATH, BASELINE_STATS_PATH, DRIFT_OUT, THRESHOLD)

        drift_detected = any(info["drift"] for info in drift_report.values())

        if drift_detected:
            logger.info("Drift detected. Retraining model...")

            # Retrain on new (drifted) data
            train_model(spark, DRIFTED_DATA_PATH, MODEL_PATH)
            print("Model retrained and logged via MLflow.")
        else:
             logger.info("No drift detected. Skipping retraining.")

    except Exception as e:
        logger.error(f"Automated retrain failed: {e}")
        sys.exit(1)

if __name__ == "__main__":

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    executor_mem = params["spark"]["executor_memory"]
    driver_mem = params["spark"]["driver_memory"]
    cores = params["spark"]["executor_cores"]

    # Start Spark session
    spark = SparkSession.builder \
        .appName("TrainTitanic") \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.cores", cores) \
        .getOrCreate()

    try:
        # Define paths
            
        # DRIFTED_DATA_PATH = "data/processed/titanic_preprocessed_drifted"
        # BASELINE_STATS_PATH = "data/baseline_stats"
        # MODEL_PATH = ""
        # THRESHOLD = 0.1

        DRIFTED_DATA_PATH = params["retrain"]["drift"]
        PROCESSED_PATH = params["retrain"]["output_data"]
        BASELINE_STATS_PATH = params["retrain"]["baseline"]
        MODEL_PATH = params["retrain"]["model_output"]
        DRIFT_OUT = params["retrain"]["output_path"]

        # Train model
        automated_retrain(spark, DRIFTED_DATA_PATH, PROCESSED_PATH, BASELINE_STATS_PATH, MODEL_PATH, DRIFT_OUT, THRESHOLD=0.1)

        print("Model training complete.")

    finally:
        spark.stop()

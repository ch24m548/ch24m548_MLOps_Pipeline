# main.py

from pyspark.sql import SparkSession
import os
from src.data_preprocessing import preprocess_data
from src.baseline_stats import get_baseline_stats
from src.train_model import train_model
from src.drift_detection import run_drift_detection
from src.simulate_drift import simulate_drift

RAW_PATH = "data/raw/titanic.csv"
PROCESSED_PATH = "data/processed/titanic_preprocessed/processed_train.parquet"
BASELINE_STATS_PATH = "data/baseline_stats"
MODEL_PATH = os.path.expanduser("~/mlops_workspace/models/logistic_model")
NEW_DATA_PATH = "data/processed/titanic_preprocessed_new"
DRIFTED_DATA_PATH = "data/processed/titanic_preprocessed_drifted"
THRESHOLD = 0.1

if __name__ == "__main__":

    spark = SparkSession.builder \
    .appName("TitanicSurvival") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    preprocess_data(spark, RAW_PATH, PROCESSED_PATH)
    print("Data preprocessing complete.")

    get_baseline_stats(spark, PROCESSED_PATH, BASELINE_STATS_PATH)
    print("Baseline statistics complete.")

    exp_id, run_id = train_model(spark, PROCESSED_PATH, MODEL_PATH)
    print("Model training complete.")

    print(f"Experiment ID and Run ID to be used while running api_server.py is: {exp_id}, {run_id}")

    # Simulate drift on the new data (can be original processed or new batch)
    simulate_drift(spark, PROCESSED_PATH, DRIFTED_DATA_PATH)
    print("Simulated drift dataset created.")

    # Run drift detection on the drifted dataset
    drift_report = run_drift_detection(spark, DRIFTED_DATA_PATH, BASELINE_STATS_PATH, THRESHOLD)
    for feature, info in drift_report.items():
        print(f"{feature}: Drift Detected = {info['drift']}, Relative Difference = {info['relative_diff']:.3f}")

    spark.stop()
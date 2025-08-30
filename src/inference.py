# src/inference.py

import os
import pandas as pd
import mlflow.pyfunc
from pyspark.sql import SparkSession
import yaml

def run_inference(spark, TEST_PARQUET_PATH, MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_STAGE, PREDICTIONS_PATH):

    df = spark.read.parquet(TEST_PARQUET_PATH).toPandas()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

    preds = model.predict(df)
    df["prediction"] = preds
    df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Inference complete. Predictions saved to: {PREDICTIONS_PATH}")

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
        
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        MODEL_NAME = os.getenv("MODEL_NAME", "TitanicSparkBestModel")
        MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
        # TEST_PARQUET_PATH = "data/processed/titanic_preprocessed/test.parquet"
        # PREDICTIONS_PATH = "data/predictions/predictions.csv"

        TEST_PARQUET_PATH = params["infer"]["test_path"]
        PREDICTIONS_PATH = params["infer"]["predict"]

        run_inference(spark, TEST_PARQUET_PATH, MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_STAGE, PREDICTIONS_PATH)

    finally:
        spark.stop()

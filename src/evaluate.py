# src/evaluate.py

import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import json
import os
import yaml

def evaluate_model(spark, model_path, test_data_path, output_metrics_path):

    # Load test data
    df = spark.read.parquet(test_data_path)

    # Load model
    model = mlflow.spark.load_model(model_path)

    # Predict
    predictions = model.transform(df)

    # Convert to pandas for metrics
    preds_pd = predictions.select("prediction", "Survived").toPandas()
    y_true = preds_pd["Survived"]
    y_pred = preds_pd["prediction"]

    # Compute metrics
    evaluator = BinaryClassificationEvaluator(labelCol="Survived")
    auc = evaluator.evaluate(predictions)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc": auc
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)

    # Save metrics
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete. Metrics saved to", output_metrics_path)


if __name__ == "__main__":

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    executor_mem = params["spark"]["executor_memory"]
    driver_mem = params["spark"]["driver_memory"]
    cores = params["spark"]["executor_cores"]

    # Start Spark session
    spark = SparkSession.builder \
        .appName("EvaluateTitanic") \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.cores", cores) \
        .getOrCreate()

    try:
    
        # Paths
        # MODEL_PATH = "models/best_model"
        # TEST_PATH = "data/processed/test_data.parquet"
        # METRICS_PATH = "report/metrics.json"
        
        MODEL_PATH = params["evaluate"]["model_output"]
        TEST_PATH = params["evaluate"]["test_path"]
        METRICS_PATH = params["evaluate"]["metrics_path"]

        # Evaluate model
        evaluate_model(spark, MODEL_PATH, TEST_PATH, METRICS_PATH)

        print("Model Evaluation complete.")

    finally:
        spark.stop()
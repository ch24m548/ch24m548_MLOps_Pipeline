# src/baseline_stats.py

from pyspark.sql import SparkSession
import yaml
import os, json
import logging
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_baseline_stats(spark, processed_data_path, output_stats_path):

    logger.info(f"Generating baseline stats from {processed_data_path}")
    
    # Load preprocessed training data
    df = spark.read.parquet(processed_data_path)
    
    # List your feature columns here - must match your training features
    feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "SexIndexed", "EmbarkedIndexed"]
    
    # Calculate descriptive statistics
    stats_df = df.select(feature_cols).describe()
    
    # Show on console for debug
    stats_df.show()
    stats_pd = stats_df.toPandas()
    
    # Save stats as JSON (or CSV if you prefer)
    # os.makedirs("data/baseline_stats", exist_ok=True)
    os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

    # whatever stats you're generating, for example:
    # stats_pd.to_json(output_stats_path, orient="records", indent=4)
    with open(output_stats_path, "w") as f:
        json.dump(stats_pd.to_dict(), f, indent=2)


if __name__ == "__main__":

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    executor_mem = params["spark"]["executor_memory"]
    driver_mem = params["spark"]["driver_memory"]
    cores = params["spark"]["executor_cores"]

    # Start Spark session
    spark = SparkSession.builder \
        .appName("BaselineTitanic") \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.cores", cores) \
        .getOrCreate()

    try:
        # Define paths
            
        # PROCESSED_PATH = "data/processed/titanic_preprocessed/processed_train.parquet"
        # BASELINE_STATS_PATH = "data/baseline_stats"
        PROCESSED_PATH = params["baseline"]["input_data"]
        BASELINE_STATS_PATH = params["baseline"]["baseline"]

        get_baseline_stats(spark, PROCESSED_PATH, BASELINE_STATS_PATH)
        print("Baseline statistics complete.")

    finally:
        spark.stop()

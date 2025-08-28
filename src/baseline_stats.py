# src/baseline_stats.py

from pyspark.sql import SparkSession
import yaml

def get_baseline_stats(spark, processed_data_path, output_stats_path):
    
    # Load preprocessed training data
    df = spark.read.parquet(processed_data_path)
    
    # List your feature columns here - must match your training features
    feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "SexIndexed", "EmbarkedIndexed"]
    
    # Calculate descriptive statistics
    stats_df = df.select(feature_cols).describe()
    
    # Show on console for debug
    stats_df.show()
    
    # Save stats as JSON (or CSV if you prefer)
    stats_df.coalesce(1).write.mode("overwrite").json(output_stats_path)


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

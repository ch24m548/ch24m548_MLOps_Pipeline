# src/simulate_drift.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import sys
import yaml

def simulate_drift(spark, input_path, output_path):

    df = spark.read.parquet(input_path)

    # Example drift simulation: Increase 'Age' by 5 years for everyone
    df_drifted = df.withColumn("Age", col("Age") + 5)

    # Another drift example: Increase Fare by 20% for passengers in Pclass 1
    df_drifted = df_drifted.withColumn(
        "Fare",
        when(col("Pclass") == 1, col("Fare") * 1.2).otherwise(col("Fare"))
    )

    # Save the drifted dataset
    df_drifted.write.mode("overwrite").parquet(output_path)

    print(f"Simulated drifted dataset saved to: {output_path}")

if __name__ == "__main__":

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    executor_mem = params["spark"]["executor_memory"]
    driver_mem = params["spark"]["driver_memory"]
    cores = params["spark"]["executor_cores"]

    # Start Spark session
    spark = SparkSession.builder \
        .appName("SimulateDriftTitanic") \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.cores", cores) \
        .getOrCreate()

    try:
        # Define paths
            
        # PROCESSED_PATH = "data/processed/titanic_preprocessed/processed_train.parquet"
        # DRIFTED_DATA_PATH = "data/processed/titanic_preprocessed_drifted"
        DRIFTED_DATA_PATH = params["simulate"]["drift"]
        PROCESSED_PATH = params["simulate"]["input_data"]

        # Simulate drift on the new data (can be original processed or new batch)
        simulate_drift(spark, PROCESSED_PATH, DRIFTED_DATA_PATH)
        print("Simulated drift dataset created.")

    finally:
        spark.stop()
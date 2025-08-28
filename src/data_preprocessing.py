# src/data_preprocessing.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
import os

import yaml

def preprocess_data(spark, input_path, output_path):

    # Load CSV
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Drop columns not useful for prediction
    df = df.drop("PassengerId", "Name", "Ticket", "Cabin")

    # Fill missing values
    mean_age = df.select("Age").dropna().agg({'Age': 'mean'}).collect()[0][0]
    df = df.fillna({'Age': mean_age})
    df = df.fillna({'Embarked': 'S'})

    # Convert categorical to numeric
    indexers = [
        StringIndexer(inputCol="Sex", outputCol="SexIndexed"),
        StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndexed")
    ]

    for indexer in indexers:
        df = indexer.fit(df).transform(df)

    # Drop original categorical columns
    df = df.drop("Sex", "Embarked")

    # Split into train and test sets (80% train, 20% test)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Save preprocessed train and test data separately
    train_path = os.path.join(output_path, "processed_train.parquet")
    test_path = os.path.join(output_path, "processed_test.parquet")
    
    train_df.write.mode("overwrite").parquet(train_path)
    test_df.write.mode("overwrite").parquet(test_path)

if __name__ == "__main__":

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    executor_mem = params["spark"]["executor_memory"]
    driver_mem = params["spark"]["driver_memory"]
    cores = params["spark"]["executor_cores"]

    # Start Spark session
    spark = SparkSession.builder \
        .appName("PreprocessTitanic") \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.cores", cores) \
        .getOrCreate()

    try:
        # Define paths
            
        # PROCESSED_PATH = "data/processed/titanic_preprocessed/processed_train.parquet"
        # RAW_PATH = "data/raw/titanic.csv"
        RAW_PATH = params["preprocess"]["raw_data"]
        PROCESSED_PATH = params["preprocess"]["output_data"]

        # Data Pre-processing
        preprocess_data(spark, RAW_PATH, PROCESSED_PATH)
        print("Data preprocessing complete.")

    finally:
        spark.stop()

# src/data_preprocessing.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer

import sys
import os
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(spark, input_path, output_path):

    logger.info(f"Starting preprocessing: raw_path={input_path}, output_path={output_path}")

    try:
            
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

        # Save preprocessed train and test data separately
        # train_path = os.path.join(output_path, "processed_train.parquet")
        # Save processed data
        df.write.mode("overwrite").parquet(output_path)

        logger.info("Preprocessing complete")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
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

        RAW_PATH = params["preprocess"]["test_data"]
        PROCESSED_PATH = params["preprocess"]["test_output"]

        # Data Pre-processing
        preprocess_data(spark, RAW_PATH, PROCESSED_PATH)
        print("Data preprocessing complete.")

    finally:
        spark.stop()

# src/data_preprocessing.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
import os

def preprocess_data(input_path, output_path):
    spark = SparkSession.builder \
        .appName("TitanicPreprocessing") \
        .getOrCreate()

    # Load CSV
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Drop columns not useful for prediction
    df = df.drop("PassengerId", "Name", "Ticket", "Cabin")

    # Fill missing values
    df = df.fillna({'Age': df.select("Age").dropna().agg({'Age': 'mean'}).collect()[0][0]})
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

    # Assemble features
    feature_cols = [col for col in df.columns if col not in ['Survived']]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df).select("features", "Survived")

    # Save preprocessed data
    df.write.mode("overwrite").parquet(output_path)

    spark.stop()

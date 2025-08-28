# test_api.py

import requests
import pandas as pd
import json
from pyspark.sql import SparkSession

# Step 1: Load sample input from your test data
spark = SparkSession.builder.appName("TestAPI").getOrCreate()
test_df = spark.read.parquet("data/processed/titanic_preprocessed/processed_test.parquet")
sample = test_df.limit(1).toPandas().drop(columns=["Survived"])  # drop label if present
spark.stop()

# Convert to JSON (dict of column:value)
payload = sample.to_dict(orient="records")[0]

# Step 2: Send request to FastAPI
url = "http://127.0.0.1:8000/predict"  # adjust if your host/port differs
response = requests.post(url, json=payload)

# Step 3: Print response
print("Input:", json.dumps(payload, indent=2))
print("Response:", response.status_code)
print("Prediction:", response.json())

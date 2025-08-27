# main.py

import os
from src.data_preprocessing import preprocess_data

RAW_PATH = "data/raw/titanic.csv"
PROCESSED_PATH = "data/processed/titanic_preprocessed"

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    preprocess_data(RAW_PATH, PROCESSED_PATH)
    print("✅ Data preprocessing complete.")

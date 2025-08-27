# main.py

import os
from src.data_preprocessing import preprocess_data
from src.train_model import train_model

RAW_PATH = "data/raw/titanic.csv"
PROCESSED_PATH = "data/processed/titanic_preprocessed"
MODEL_PATH = "models/logistic_model"

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    preprocess_data(RAW_PATH, PROCESSED_PATH)
    print("✅ Data preprocessing complete.")

    train_model(PROCESSED_PATH, MODEL_PATH)
    print("✅ Model training complete.")

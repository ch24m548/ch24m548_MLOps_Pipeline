# app/main.py

from fastapi import FastAPI
from app.schema import TitanicPassenger
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Titanic Survival Predictor")

# Load model from MLflow Registry
MODEL_NAME = "TitanicSparkModel"
STAGE = "Staging"  # or "Production"

print("🔄 Loading model from MLflow Registry...")
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{STAGE}"
)
print("✅ Model loaded.")

@app.get("/")
def root():
    return {"message": "Titanic prediction API is up!"}

@app.post("/predict")
def predict(passenger: TitanicPassenger):
    input_data = pd.DataFrame([passenger.dict()])
    prediction = model.predict(input_data)
    return {
        "prediction": int(prediction[0]),
        "survived": "Yes" if prediction[0] == 1 else "No"
    }

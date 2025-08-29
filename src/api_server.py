# src/api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import uvicorn
import logging
import pandas as pd

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the expected input data model (match your features)
class PassengerFeatures(BaseModel):
    Pclass: float
    Age: float
    SibSp: float
    Parch: float
    Fare: float
    SexIndexed: float
    EmbarkedIndexed: float

app = FastAPI()

# Load model from MLflow Model Registry
model_name = "TitanicSparkModel"
model_stage = "Production"  # or "Staging"

try:
    logger.info(f"Loading model '{model_name}' from stage '{model_stage}'...")
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
    # model = mlflow.pyfunc.load_model("mlruns/792831802516628706/d90b8f58d5ec4693bfdf33bdd8db5720/artifacts/model")
    logger.info("Model loaded successfully.")

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

@app.post("/predict")
def predict(features: PassengerFeatures):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dataframe for model
        input_dict = features.dict()
        input_df = pd.DataFrame([input_dict])
            
        # Get prediction
        prediction = model.predict(input_df)

        # Return the prediction result (assuming binary classification: 0 or 1)
        return {"prediction": int(prediction[0])}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

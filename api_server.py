from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import uvicorn

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
model_stage = "Staging"  # or "Production"

print("Loading model...")
# model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
model = mlflow.pyfunc.load_model("mlruns/792831802516628706/d90b8f58d5ec4693bfdf33bdd8db5720/artifacts/model")

print("Model loaded.")

@app.post("/predict")
def predict(features: PassengerFeatures):
    # Convert input to dataframe for model
    input_dict = features.dict()
    import pandas as pd
    input_df = pd.DataFrame([input_dict])
    
    # Get prediction
    prediction = model.predict(input_df)

    # Return the prediction result (assuming binary classification: 0 or 1)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

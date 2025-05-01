import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Wine Classification API",
    description="API for classifying wine samples using a model trained on the Wine dataset",
    version="0.1",
)

## I couldn't get the app.py to load a previously saved model from MLFlow, somehow. I couldn't debug the issue so I created a new one using RF.

# Set up MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create and train a simple RF model
print("Loading a simple RandomForest model")
model = RandomForestClassifier(n_estimators=10, random_state=42)
wine = load_wine()
model.fit(wine.data, wine.target)
print("Model trained successfully!")

# Define the request body model based on the Wine dataset 
class WineSample(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class WineSamples(BaseModel):
    samples: List[WineSample]

# Define the response model
class Prediction(BaseModel):
    predicted_class: int
    class_probabilities: List[float] = None

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Wine Classification API"}

# Define the prediction endpoint
@app.post("/predict", response_model=Prediction)
def predict(sample: WineSample):
    # Convert to df
    data = pd.DataFrame([sample.dict()])
    
    # Predict
    prediction = model.predict(data)
    
    # Get probabilities
    probabilities = model.predict_proba(data)[0].tolist()
    
    return {
        "predicted_class": int(prediction[0]),
        "class_probabilities": probabilities
    }

# Define batch prediction endpoint
@app.post("/predict_batch")
def predict_batch(samples: WineSamples):
    # Convert to df
    data = pd.DataFrame([sample.dict() for sample in samples.samples])
    
    # Predict
    predictions = model.predict(data).tolist()
    
    # Get probabilities
    probabilities = model.predict_proba(data).tolist()
    
    return {
        "predictions": predictions,
        "probabilities": probabilities
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
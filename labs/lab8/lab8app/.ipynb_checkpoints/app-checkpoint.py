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

# In app.py, replace the model loading section with this:

# Set up MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Try to load the model using the specific run ID
run_id = "8d49265100154886a0268a32613fc6ad"
try:
    # First attempt: try the standard path for sklearn models
    model_path = f"runs:/{run_id}/sklearn-model"
    model = mlflow.sklearn.load_model(model_path)
    print(f"Successfully loaded sklearn model from run {run_id}")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    try:
        # Second attempt: try another common path
        model_path = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_path)
        print(f"Successfully loaded model from run {run_id}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        try:
            # Third attempt: check for xgboost model
            model_path = f"runs:/{run_id}/xgboost_model"
            model = mlflow.sklearn.load_model(model_path)
            print(f"Successfully loaded xgboost model from run {run_id}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            try:
                # Fourth attempt: try decision tree model
                model_path = f"runs:/{run_id}/decision_tree_model"
                model = mlflow.sklearn.load_model(model_path)
                print(f"Successfully loaded decision tree model from run {run_id}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                try:
                    # Fifth attempt: try random forest model
                    model_path = f"runs:/{run_id}/random_forest_model"
                    model = mlflow.sklearn.load_model(model_path)
                    print(f"Successfully loaded random forest model from run {run_id}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    # Final fallback: train a simple model
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.datasets import load_wine
                    print("All model loading attempts failed. Loading a simple default model instead")
                    model = RandomForestClassifier(n_estimators=10, random_state=42)
                    wine = load_wine()
                    model.fit(wine.data, wine.target)

# Define the request body model
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
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([sample.dict()])
        
        # Make prediction
        prediction = model.predict(data)
        
        # Get class probabilities if model supports it
        probabilities = None
        try:
            probabilities = model.predict_proba(data)[0].tolist()
        except:
            pass
        
        return {
            "predicted_class": int(prediction[0]),
            "class_probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Define batch prediction endpoint
@app.post("/predict_batch")
def predict_batch(samples: WineSamples):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([sample.dict() for sample in samples.samples])
        
        # Make predictions
        predictions = model.predict(data).tolist()
        
        # Get class probabilities if model supports it
        probabilities = None
        try:
            probabilities = model.predict_proba(data).tolist()
        except:
            pass
        
        return {
            "predictions": predictions,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
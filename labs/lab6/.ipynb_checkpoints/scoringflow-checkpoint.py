"""
Super Simple Scoring Flow for ML Model
This flow loads a model from MLFlow registry and makes predictions.
"""
from metaflow import FlowSpec, step
import pandas as pd
import mlflow
import os

class SimpleScoringFlow(FlowSpec):
    
    @step
    def start(self):
        """
        Starting point: Get data for scoring
        """
        print("Starting the scoring flow...")
        
        # Set up MLFlow
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Load sample data for scoring (using wine dataset as fallback)
        from sklearn import datasets
        wine = datasets.load_wine()
        self.data = pd.DataFrame(wine.data[:20], columns=wine.feature_names)
        print(f"Using sample wine dataset: {self.data.shape[0]} rows")
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """
        Load the trained model from MLFlow
        """
        print("Loading model from MLFlow...")
        
        try:
            # Load latest model version by name
            model_name = "metaflow-rf-model"
            model_uri = f"models:/{model_name}/latest"
            self.model = mlflow.sklearn.load_model(model_uri)
            print(f"Loaded latest version of model: {model_name}")
        except Exception as e:
            print(f"Error loading model from MLFlow: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
        
        self.next(self.predict)
    
    @step
    def predict(self):
        """
        Make predictions using the loaded model
        """
        print("Making predictions...")
        
        # Make predictions
        self.predictions = self.model.predict(self.data)
        print(f"Made predictions for {len(self.predictions)} samples")
        
        # Create a results DataFrame
        self.results = pd.DataFrame()
        self.results['prediction'] = self.predictions
        
        # Add the features for reference
        for col in self.data.columns:
            self.results[f'feature_{col}'] = self.data[col]
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Output the predictions
        """
        # Save results to CSV
        output_path = "predictions.csv"
        self.results.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
        
        # Print summary
        print("Scoring flow completed successfully!")
        print(f"Generated {len(self.predictions)} predictions")
        print(f"Prediction distribution: {pd.Series(self.predictions).value_counts().to_dict()}")
        print("First few predictions:")
        print(self.results[['prediction']].head())


if __name__ == '__main__':
    SimpleScoringFlow()
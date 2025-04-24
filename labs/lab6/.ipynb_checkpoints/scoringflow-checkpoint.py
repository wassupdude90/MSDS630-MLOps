"""
Scoring Flow for ML Model
This flow handles loading a trained model from MLFlow registry
and making predictions on new data.
"""
from metaflow import FlowSpec, step, Parameter, Flow
import pandas as pd
import mlflow
import os

class ScoringFlow(FlowSpec):
    # Parameters
    model_name = Parameter('model_name',
                         help='Name of the registered model in MLFlow',
                         default='metaflow-rf-model')
    
    data_path = Parameter('data_path',
                        help='Path to the data file for scoring',
                        default=None)
    
    run_id = Parameter('run_id',
                      help='MLFlow run ID of the model to use',
                      default=None)
    
    @step
    def start(self):
        """
        Starting point: Load the model and data
        """
        print("Starting the scoring flow...")
        
        # Set up MLFlow
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Get the latest training flow run for reference
        if self.run_id is None:
            try:
                train_run = Flow('TrainingFlow').latest_run
                self.run_id = train_run.data.run_id
                print(f"Using latest training flow run ID: {self.run_id}")
            except Exception as e:
                print(f"Could not get latest training flow run: {e}")
                print("Please provide a valid run_id parameter")
                print("Using latest model version instead")
                self.run_id = None
        
        # Load data for scoring
        if self.data_path is not None:
            try:
                self.data = pd.read_csv(self.data_path)
                print(f"Loaded scoring data from {self.data_path}: {self.data.shape[0]} rows")
            except Exception as e:
                print(f"Error loading data from {self.data_path}: {e}")
                self.fallback_data()
        else:
            self.fallback_data()
        
        self.next(self.preprocess)
    
    def fallback_data(self):
        """Helper function to load fallback data when provided data is not available"""
        print("Loading fallback data...")
        
        try:
            # Try to get test data from the latest training flow
            train_run = Flow('TrainingFlow').latest_run
            self.data = pd.DataFrame(train_run['preprocess'].task.data.X_test)
            print(f"Using test data from latest training flow: {self.data.shape[0]} rows")
        except:
            # If that fails, use a sample dataset
            from sklearn import datasets
            wine = datasets.load_wine()
            self.data = pd.DataFrame(wine.data[:20], columns=wine.feature_names)
            print(f"Using sample wine dataset: {self.data.shape[0]} rows")
    
    @step
    def preprocess(self):
        """
        Preprocess the data for scoring
        """
        print("Preprocessing scoring data...")
        
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())
        
        # Make sure the data is ready for prediction
        # (For a real application, ensure the same preprocessing as during training)
        self.X = self.data
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """
        Load the trained model from MLFlow
        """
        print(f"Loading model {self.model_name} from MLFlow...")
        
        try:
            if self.run_id:
                # Load model from specific run
                self.model = mlflow.sklearn.load_model(f"runs:/{self.run_id}/model")
                print(f"Loaded model from run ID: {self.run_id}")
            else:
                # Load latest model version
                self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/latest")
                print(f"Loaded latest version of model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback: Try to get model from the latest training flow
            try:
                train_run = Flow('TrainingFlow').latest_run
                self.model = train_run['select_best_model'].task.data.best_model['model']
                print("Using model from latest training flow")
            except Exception as e2:
                print(f"Could not load model from latest training flow: {e2}")
                raise RuntimeError("Could not load any model for scoring")
        
        self.next(self.predict)
    
    @step
    def predict(self):
        """
        Make predictions using the loaded model
        """
        print("Making predictions...")
        
        # Generate predictions
        self.predictions = self.model.predict(self.X)
        
        # Generate probabilities if available
        try:
            self.probabilities = self.model.predict_proba(self.X)
            self.has_probabilities = True
        except:
            self.has_probabilities = False
        
        print(f"Generated predictions for {len(self.predictions)} samples")
        
        self.next(self.output_results)
    
    @step
    def output_results(self):
        """
        Format and output the prediction results
        """
        # Create output DataFrame
        results = pd.DataFrame()
        results['prediction'] = self.predictions
        
        # Add probabilities if available
        if self.has_probabilities:
            for i in range(self.probabilities.shape[1]):
                results[f'probability_class_{i}'] = self.probabilities[:, i]
        
        # Add original features for reference
        for col in self.X.columns:
            results[f'feature_{col}'] = self.X[col]
        
        # Save results to CSV
        output_path = "predictions.csv"
        results.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
        
        # Store the results for the end step
        self.results = results
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End of the flow
        """
        print("Scoring flow completed successfully!")
        print(f"Generated {len(self.predictions)} predictions")
        print(f"Prediction distribution: {pd.Series(self.predictions).value_counts().to_dict()}")
        print("First few predictions:")
        print(self.results[['prediction'] + 
                          (['probability_class_0', 'probability_class_1', 'probability_class_2'] 
                           if self.has_probabilities else [])].head())

if __name__ == '__main__':
    ScoringFlow()
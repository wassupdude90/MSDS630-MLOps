from metaflow import FlowSpec, step, Parameter, Flow
import pandas as pd
import mlflow
import os

## Scoring Flow handles loading the trained model from MLFlow registry and making predictions on new data
class ScoringFlow(FlowSpec):
    # Params
    model_name = Parameter('model_name', help='Name of the registered model in MLFlow', default='metaflow-rf-model')
    data_path = Parameter('data_path', help='Path to the data file for scoring', default=None)
    run_id = Parameter('run_id', help='MLFlow run ID of the model to use', default=None)
    
    @step
    def start(self):
        print("Starting the scoring flow...")
        
        # Setup
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Get the latest training flow
        try:
            if self.run_id is None:
                train_run = Flow('TrainingFlow').latest_run
                self.mlflow_run_id = train_run.data.run_id
                print(f"Using latest training flow run ID: {self.mlflow_run_id}")
            else:
                # Use the given run_id parameter, if provided; but I'm just running the latest train flow though
                self.mlflow_run_id = self.run_id
        except Exception as e:
            print(f"Could not get latest training flow run: {e}")
            print("Using latest model version instead")
            self.mlflow_run_id = None
        
        # Load data for scoring
        if self.data_path is not None:
            self.data = pd.read_csv(self.data_path)
            print(f"Loaded scoring data from {self.data_path}: {self.data.shape[0]} rows")
        else:
            self.fallback_data()
        
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """
        Preprocess the data for scoring
        """
        print("Preprocessing scoring data...")
        
        # Handle null values
        self.data = self.data.fillna(self.data.mean())
        
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
                # Load model from specific run, if I want to
                self.model = mlflow.sklearn.load_model(f"runs:/{self.run_id}/model")
                print(f"Loaded model from run ID: {self.run_id}")
            else:
                # Load latest model version
                self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/latest")
                print(f"Loaded latest version of model: {self.model_name}")        
        self.next(self.predict)
    
    @step
    def predict(self):
        """
        Make predictions using the loaded model
        """
        print("Making predictions...")
        
        # Generate predictions
        self.predictions = self.model.predict(self.X)
        
        # Generate probabilities, if available
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
        Format and save the predictions
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
        
        # Save results \
        output_path = "predictions.csv"
        results.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
        
        # Store the results for the end step
        self.results = results
        
        self.next(self.end)
    
    @step
    def end(self):
        print("Scoring flow done!")
        print(f"Generated {len(self.predictions)} predictions")
        print(f"Prediction distribution: {pd.Series(self.predictions).value_counts().to_dict()}")
        print("First few prdeictions:")
        print(self.results[['prediction'] + 
                          (['probability_class_0', 'probability_class_1', 'probability_class_2'] 
                           if self.has_probabilities else [])].head())

if __name__ == '__main__':
    ScoringFlow()
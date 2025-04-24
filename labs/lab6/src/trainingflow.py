from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

## Training Flow to handle data loading, preprocessing, model training, and model registration 
class TrainingFlow(FlowSpec):
    # Params
    random_state = Parameter('random_state', help='Random seed for reproducibility', default=42, type=int)
    test_size = Parameter('test_size', help='Proportion of data to use for testing', default=0.2, type=float)
    model_type = Parameter('model_type', help='Type of model to train (rf, xgb)', default='rf')
    
    @step
    def start(self):
        print("Starting the training flow...")
        
        # Setup
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("metaflow-training")

        # Load the Wine dataset, my favorite lol
        wine = datasets.load_wine()
        self.data = pd.DataFrame(wine.data, columns=wine.feature_names)
        self.data['target'] = wine.target
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """
        Preprocessing the data and performing feature engineering
        """
        print("Preprocessing data...")
        
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())
        
        # Define features/target
        if 'target' in self.data.columns:
            self.target_col = 'target'
        else:
            # Last column is the target
            self.target_col = self.data.columns[-1]
        
        # Split features/target
        self.X = self.data.drop(self.target_col, axis=1)
        self.y = self.data[self.target_col]
        
        # Feature names
        self.feature_names = list(self.X.columns)
        
        # Train-test split        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        self.next(self.train_models)
    
    @step
    def train_models(self):
        """
        Train using RF and XGB with different hyperparameters
        """
        print(f"Training {self.model_type} models with different hyperparameters...")
        
        self.models = []
        self.scores = []
        
        # Train different models based on model_type 
        if self.model_type == 'rf':
            
            # Define hyperparameters; just trying a few
            n_estimators_list = [100, 200]
            max_depth_list = [None, 10, 20]
            
            # Train models with different hyperparameters
            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=self.random_state
                    )
                    
                    # Train model
                    model.fit(self.X_train, self.y_train)
                    
                    # Evaluate model
                    score = model.score(self.X_test, self.y_test)
                    
                    # Store model and score
                    self.models.append({
                        'model': model,
                        'params': {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth
                        },
                        'score': score
                    })
                    self.scores.append(score)
                    
                    print(f"Model with n_estimators={n_estimators}, max_depth={max_depth}: score={score:.4f}")
        
        elif self.model_type == 'xgb':
            try:                                
                # Define hyperparameters
                n_estimators_list = [100, 200]
                max_depth_list = [3, 6, 9]
                
                # Train models with different hyperparameters
                for n_estimators in n_estimators_list:
                    for max_depth in max_depth_list:
                        model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=self.random_state
                        )
                        
                        # Train model
                        model.fit(self.X_train, self.y_train)
                        
                        # Evaluate model
                        score = model.score(self.X_test, self.y_test)
                        
                        # Store model and score
                        self.models.append({
                            'model': model,
                            'params': {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth
                            },
                            'score': score
                        })
                        self.scores.append(score)
                        
                        print(f"Model with n_estimators={n_estimators}, max_depth={max_depth}: score={score:.4f}")
                self.next(self.train_models)
        
        self.next(self.select_best_model)
    
    @step
    def select_best_model(self):
        """
        Select the best model based on test score
        """
        # Find the best model
        best_idx = np.argmax(self.scores)
        self.best_model = self.models[best_idx]
        
        print(f"Best model: {self.model_type} with params {self.best_model['params']}")
        print(f"Best score: {self.best_model['score']:.4f}")
        
        self.next(self.register_model)
    
    @step
    def register_model(self):
        """
        Register the best model with MLFlow
        """
        print("Registering best model with MLFlow...")
        
        # Start an MLFlow run (pretty easy from here on)
        with mlflow.start_run() as run:
            # Log parameters
            for param_name, param_value in self.best_model['params'].items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            mlflow.log_metric("accuracy", self.best_model['score'])
            
            # Log model
            model_name = f"metaflow-{self.model_type}-model"
            mlflow.sklearn.log_model(
                self.best_model['model'], 
                artifact_path="model",
                registered_model_name=model_name
            )
            
            # Store the run ID for reference
            self.run_id = run.info.run_id
            
            print(f"Model registered with MLFlow as {model_name}")
            print(f"MLFlow run ID: {self.run_id}")
        
        self.next(self.end)
    
    @step
    def end(self):
        print("Training flow done!")
        print(f"Best model ({self.model_type}) accuracy: {self.best_model['score']:.4f}")
        print(f"Model parameters: {self.best_model['params']}")
        print(f"MLFlow run ID: {self.run_id}")

if __name__ == '__main__':
    TrainingFlow()
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import os

# Create output directory (it doesn't exist at the beginning)
os.makedirs('data/processed', exist_ok=True)

# Load the data
wine = load_wine() # Just getting the Wine data from the source
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['target'] = wine.target

# Save raw data
data.to_csv('data/wine_data.csv', index=False)

# Preprocessing steps
def preprocess_data():
    # Split X & Y
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Scale 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create processed df
    processed_data = pd.DataFrame(X_scaled, columns=X.columns)
    processed_data['target'] = y
    
    # Save processed data
    processed_data.to_csv('data/processed/wine_scaled.csv', index=False)
    
    print("Preprocessing completed successfully") # Let's me know that code ran successfully
    return processed_data

if __name__ == "__main__":
    processed_data = preprocess_data()
    print(f"Processed data shape: {processed_data.shape}") # Print data shape
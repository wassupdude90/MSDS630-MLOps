import requests
import json

# API local URL
url = 'http://127.0.0.1:8000/predict'

# Sample data for a wine (using standard values from the Wine dataset)
sample_data = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.80,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315_of_diluted_wines": 3.92,
    "proline": 1065.0
}

# Request to the API
response = requests.post(url, json=sample_data)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
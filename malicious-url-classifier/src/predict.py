import joblib
import numpy as np
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler

def extract_features(url):
    # Example feature extraction: length of the URL and number of dots
    parsed_url = urlparse(url)
    features = [
        len(url),  # Length of the URL
        url.count('.'),  # Number of dots in the URL
        len(parsed_url.path),  # Length of the path
        len(parsed_url.query),  # Length of the query
    ]
    return np.array(features).reshape(1, -1)

def predict_url(url):
    # Load the trained model
    model = joblib.load("models/model.pkl")
    
    # Extract features from the input URL
    features = extract_features(url)
    
    # Standardize features (if the model was trained on standardized data)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # Adjust this if you saved the scaler during training
    
    # Make prediction
    prediction = model.predict(features)
    
    return "Malicious" if prediction[0] == 1 else "Benign"

if __name__ == "__main__":
    user_input = input("Enter a URL to classify: ")
    result = predict_url(user_input)
    print(f"The URL '{user_input}' is classified as: {result}")
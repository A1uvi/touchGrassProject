import pandas as pd
import joblib
import json
from sklearn.preprocessing import LabelEncoder

# Load the encoding mappings
with open('static/assets/encoding_mappings.json', 'r') as f:
    encoding_mappings = json.load(f)

# Load the trained Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

def encode_input_data(input_data):
    # Convert input_data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for column in input_df.columns:
        if column in encoding_mappings:
            le = LabelEncoder()
            le.classes_ = np.array(list(encoding_mappings[column].keys()))
            input_df[column] = le.transform(input_df[column])
    
    return input_df

def predict_sustainability_index(input_data):
    # Encode the input data
    encoded_input = encode_input_data(input_data)
    
    # Make predictions using the model
    prediction = rf_model.predict(encoded_input)
    return prediction[0]

# Example input data (replace with actual values)
input_data = {
    'Feature1': 'value1',  # Replace with actual feature names and values
    'Feature2': 'value2',
    'Feature3': 'value3',
    # Add all necessary features here
}

# Predict sustainability index
predicted_index = predict_sustainability_index(input_data)
print(f'Predicted Sustainability Index: {predicted_index}')

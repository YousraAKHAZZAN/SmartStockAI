from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the model and scaler
with open('lr_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature order must match exactly your training data
COLUMNS_ORDER = [
    'Category',
    'Region',
    'Inventory',
    'Sales',
    'Orders',
    'Price',
    'Discount',
    'Weather',
    'Promotion',
    'Competitor Price',
    'Seasonality'
]

# Value ranges based on your dataset
FEATURE_RANGES = {
    'Category': (1, 4),
    'Region': (0, 3),
    'Inventory': (0, 500),
    'Sales': (0, 200),
    'Orders': (0, 200),
    'Price': (0, 100),
    'Discount': (0, 30),
    'Weather': (0, 3),
    'Promotion': (0, 1),
    'Competitor Price': (0, 100),
    'Seasonality': (0, 3)
}

def validate_data(data):
    errors = []
    for field in COLUMNS_ORDER:
        if field not in data:
            errors.append(f"Missing field: {field}")
        elif field in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[field]
            if data[field] < min_val or data[field] > max_val:
                errors.append(f"{field} should be between {min_val} and {max_val}")
    return errors

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        errors = validate_data(data)
        if errors:
            return jsonify({"errors": errors}), 400
        
        # Prepare features in correct order
        features = []
        for col in COLUMNS_ORDER:
            features.append(data[col])
            
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction and scale it
        raw_prediction = model.predict(features_scaled)[0]
        scaled_prediction = raw_prediction / 100
        
        return jsonify({
            "demand_forecast": float(scaled_prediction),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
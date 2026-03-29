# app.py — Main Flask Application
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
 
# Initialize Flask app
app = Flask(__name__)
 
# Load the saved model, scaler, and features (from Block 3)
MODEL_PATH   = os.path.join('..', 'models', 'best_model.pkl')
SCALER_PATH  = os.path.join('..', 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join('..', 'models', 'features.pkl')
 
model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
 
print('Model loaded successfully!')
print(f'Expected features: {features}')
 
# -----------------------------------------------
# ROUTE 1: Homepage — serves the HTML form
# -----------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')
 
# -----------------------------------------------
# ROUTE 2: Prediction endpoint
# -----------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
 
        # Build input dataframe
        input_df = pd.DataFrame([{
            'Store_Type_encoded':    int(data.get('Store_Type_encoded', 0)),
            'Location_Type_encoded': int(data.get('Location_Type_encoded', 0)),
            'Region_Code_encoded':   int(data.get('Region_Code_encoded', 0)),
            'Holiday':   int(data.get('Holiday', 0)),
            'Discount':  int(data.get('Discount', 0)),
            'Year':      int(data.get('Year', 2022)),
            'Month':     int(data.get('Month', 1)),
            'Day':       int(data.get('Day', 1)),
            'DayOfWeek': int(data.get('DayOfWeek', 0)),
            'WeekOfYear': int(data.get('WeekOfYear', 1)),
            'IsWeekend':     int(data.get('IsWeekend', 0)),
            'IsMonthStart':  int(data.get('IsMonthStart', 0)),
            'IsMonthEnd':    int(data.get('IsMonthEnd', 0)),
            'Quarter': int(data.get('Quarter', 1)),
            'Season':  int(data.get('Season', 1)),
            '#Order':  int(data.get('Order', 50))
        }])
 
        # Make prediction
        prediction = model.predict(input_df[features])[0]
 
        return jsonify({
            'status': 'success',
            'predicted_sales': round(float(prediction), 2),
            'message': f'Predicted sales: {prediction:.2f}'
        })
 
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
 
# -----------------------------------------------
# ROUTE 3: Health check (confirms server is running)
# -----------------------------------------------
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Server is running!'})
 
# -----------------------------------------------
# Run the app
# -----------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

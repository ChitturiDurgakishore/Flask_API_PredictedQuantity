import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import os

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
MODEL_DIR = 'mediguide_ml_model'
MODEL_FILENAME = 'mediguide_stock_predictor.joblib'
FEATURES_FILENAME = 'model_features.joblib'

# --- Load Model and Features ---
try:
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    features_path = os.path.join(MODEL_DIR, FEATURES_FILENAME)
    
    model = joblib.load(model_path)
    model_features = joblib.load(features_path)
    print("ML model and features loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or features file not found in '{MODEL_DIR}'. Please run train_model.py first.")
    exit()
except Exception as e:
    print(f"Error loading model or features: {e}")
    exit()

# --- Helper Function for Feature Engineering (MUST match train_model.py) ---
def prepare_input_data(input_json, historical_data_df=None):
    """
    Prepares input data for prediction, matching the feature engineering
    done during model training.

    Args:
        input_json (dict): Dictionary with new data (e.g., from API request).
                           Expected keys: 'pharmacy_id', 'medicine_name', 
                           'date' (YYYY-MM-DD format), 'price_at_sale', 
                           'stock_before_sale', 'weather_condition', 
                           'pharmacy_area'.
        historical_data_df (pd.DataFrame, optional): A DataFrame of relevant
                           historical sales data for lag and rolling features.
                           In a real app, this would come from your DB.
                           For this project, it's simplified.
    Returns:
        pd.DataFrame: A DataFrame with features ready for prediction.
    """
    
    # Create a DataFrame for the new input
    input_df = pd.DataFrame([input_json])
    
    # Convert 'date' to datetime and rename to 'sold_at' for consistency with training
    input_df['sold_at'] = pd.to_datetime(input_df['date'])
    input_df = input_df.drop(columns=['date'])

    # Ensure correct data types for new input
    input_df['pharmacy_id'] = input_df['pharmacy_id'].astype(str) # Match training
    input_df['price_at_sale'] = pd.to_numeric(input_df['price_at_sale'])
    input_df['stock_before_sale'] = pd.to_numeric(input_df['stock_before_sale'])
    
    # --- Feature Engineering (MUST match train_model.py) ---
    # Time-based features
    input_df['day_of_year'] = input_df['sold_at'].dt.dayofyear
    input_df['month'] = input_df['sold_at'].dt.month
    input_df['year'] = input_df['sold_at'].dt.year
    input_df['week_of_year'] = input_df['sold_at'].dt.isocalendar().week.astype(int)
    input_df['day_of_month'] = input_df['sold_at'].dt.day
    input_df['is_weekend'] = ((input_df['sold_at'].dt.dayofweek == 5) | (input_df['sold_at'].dt.dayofweek == 6)).astype(int)
    
    # Derive day_of_week and season for the new input
    day_of_week_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    seasons_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Summer', 6: 'Summer',
        7: 'Summer', 8: 'Monsoon', 9: 'Monsoon', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'
    }
    input_df['day_of_week'] = input_df['sold_at'].dt.dayofweek.map(lambda x: day_of_week_names[x])
    input_df['season'] = input_df['sold_at'].dt.month.map(seasons_map)

    # --- Lag and Rolling Features (Simplification for API Demo) ---
    # In a real-world scenario, you would query your 'pharmacies_sales' database
    # for historical 'quantity_sold' data specific to the pharmacy_id and medicine_name
    # for the relevant past periods to accurately calculate these features.
    # For this project demo, we'll set them to 0 or a placeholder.
    # This simplification means the model relies less on time-series patterns for live predictions.
    
    input_df['quantity_sold_lag_1'] = 0 
    input_df['quantity_sold_lag_7'] = 0
    input_df['quantity_sold_rolling_mean_7d'] = 0.0
    input_df['quantity_sold_rolling_std_7d'] = 0.0

    # --- One-Hot Encoding for Categorical Features ---
    # Apply one-hot encoding consistent with training data.
    # This requires ensuring all possible categories from training are handled.
    # The 'model_features' list contains the column names generated during training.
    
    # Initialize a DataFrame with all expected model features, filled with zeros
    processed_df = pd.DataFrame(0, index=input_df.index, columns=model_features)

    # Copy over the numeric features
    numeric_cols = [
        'price_at_sale', 'stock_before_sale', 'day_of_year', 'month', 'year',
        'week_of_year', 'day_of_month', 'is_weekend',
        'quantity_sold_lag_1', 'quantity_sold_lag_7',
        'quantity_sold_rolling_mean_7d', 'quantity_sold_rolling_std_7d'
    ]
    for col in numeric_cols:
        if col in processed_df.columns and col in input_df.columns:
            processed_df[col] = input_df[col]

    # Handle one-hot encoded categorical features
    # Ensure columns match what's in model_features (which include 'pharmacy_id_XXX', 'medicine_name_XXX' etc.)
    for _, row in input_df.iterrows():
        # pharmacy_id
        col_name = f"pharmacy_id_{row['pharmacy_id']}"
        if col_name in processed_df.columns:
            processed_df.loc[processed_df.index == _ , col_name] = 1
        
        # medicine_name
        col_name = f"medicine_name_{row['medicine_name']}"
        if col_name in processed_df.columns:
            processed_df.loc[processed_df.index == _ , col_name] = 1

        # day_of_week
        col_name = f"day_of_week_{row['day_of_week']}"
        if col_name in processed_df.columns:
            processed_df.loc[processed_df.index == _ , col_name] = 1

        # season
        col_name = f"season_{row['season']}"
        if col_name in processed_df.columns:
            processed_df.loc[processed_df.index == _ , col_name] = 1

        # weather_condition
        col_name = f"weather_condition_{row['weather_condition']}"
        if col_name in processed_df.columns:
            processed_df.loc[processed_df.index == _ , col_name] = 1

        # pharmacy_area (if included)
        col_name = f"pharmacy_area_{row['pharmacy_area']}"
        if col_name in processed_df.columns:
            processed_df.loc[processed_df.index == _ , col_name] = 1
    
    # Drop 'id' and 'sold_at' columns if they were included in the input_df, 
    # as they are not model features themselves, but used for feature engineering.
    processed_df = processed_df.drop(columns=[col for col in ['id', 'sold_at'] if col in processed_df.columns], errors='ignore')

    # Ensure the order of columns matches the model's expected features
    processed_df = processed_df[model_features]

    return processed_df


# --- Prediction Endpoint ---
@app.route('/predict_quantity', methods=['POST'])
def predict_quantity():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Define required fields for the new model's input
    required_fields = [
        'pharmacy_id', 'medicine_name', 'date', 'price_at_sale', 
        'stock_before_sale', 'weather_condition', 'pharmacy_area'
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    
    try:
        # Prepare the input data using the helper function
        # For a full implementation, historical_data_df would be fetched from your DB
        processed_input = prepare_input_data(data)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        
        # Ensure prediction is not negative (sales cannot be negative)
        prediction = max(0, int(round(prediction)))

        return jsonify({"predicted_quantity": prediction})

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during prediction."}), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running and model loaded."}), 200

# --- Run the Flask App ---
if __name__ == '__main__':
    # It's recommended to run Flask in production with a WSGI server like Gunicorn or uWSGI
    # For local development, you can run it directly:
    app.run(debug=True, port=5000)
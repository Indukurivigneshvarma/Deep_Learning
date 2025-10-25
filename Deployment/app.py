from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved Keras model
model = tf.keras.models.load_model('California_Model.keras')

# Load saved scaler
scaler = joblib.load('scaler.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input features from form
        input_features = [float(x) for x in request.form.values()]

        # Convert to DataFrame with same columns as scaler
        features_df = pd.DataFrame([input_features], columns=scaler.feature_names_in_)

        # Scale features
        scaled_features = scaler.transform(features_df)

        # Predict
        prediction = model.predict(scaled_features)[0][0]

        # Send result to HTML
        return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction:,.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)



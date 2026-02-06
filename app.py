import joblib

imputer = joblib.load('imputer.joblib')

scaler = joblib.load('scaler.joblib')

model = joblib.load('logistic_regression_model.joblib')

print("Imputer, Scaler, and Model loaded successfully within app.py")

from flask import Flask, request, jsonify

app = Flask(__name__)

print("Flask application initialized and models loaded in app.py")

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    data = request.get_json()
    print(f"Received data for prediction: {data}")
    # Further processing of 'data' will be done in subsequent steps
    return jsonify({"message": "Data received successfully"})

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

print("Flask application initialized and models loaded in app.py")

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    data = request.get_json()
    print(f"Received data for prediction: {data}")

    try:
        input_df = pd.DataFrame([data], columns=feature_columns)
        
        columns_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        input_df[columns_to_replace_zero] = input_df[columns_to_replace_zero].replace(0, np.nan)

        imputed_data = imputer.transform(input_df)
        imputed_df = pd.DataFrame(imputed_data, columns=feature_columns, index=input_df.index)

        scaled_data = scaler.transform(imputed_df)
        scaled_df = pd.DataFrame(scaled_data, columns=feature_columns, index=input_df.index)

        prediction = model.predict(scaled_df)
        prediction_proba = model.predict_proba(scaled_df)

        predicted_class = int(prediction[0])
        probability = prediction_proba[0][predicted_class]

        print(f"Raw prediction: {prediction[0]}")
        print(f"Prediction probability: {prediction_proba[0]}")

        return jsonify({
            "prediction": predicted_class,
            "probability": float(probability),
            "message": "Prediction successful"
        })

    except Exception as e:
        return jsonify({"error": str(e), "message": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
    print("Flask development server is running.")



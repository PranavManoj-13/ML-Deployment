import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

imputer = joblib.load("imputer.joblib")
scaler = joblib.load("scaler.joblib")
model = joblib.load("logistic_regression_model.joblib")

feature_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json(force=True)

        # Validate input
        missing = set(feature_columns) - set(data.keys())
        if missing:
            return jsonify({"error": f"Missing fields: {list(missing)}"}), 400

        input_df = pd.DataFrame([data], columns=feature_columns)

        # Replace invalid zeros
        zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        input_df[zero_as_nan] = input_df[zero_as_nan].replace(0, np.nan)

        # Preprocessing
        imputed = imputer.transform(input_df)
        scaled = scaler.transform(imputed)

        # Prediction
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][pred]

        return jsonify(
            prediction=int(pred),
            probability=float(proba),
        )

    except Exception as e:
        return jsonify(error=str(e)), 500

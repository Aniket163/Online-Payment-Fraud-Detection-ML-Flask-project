from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
MODEL_PATH = 'model.joblib'
PREP_PATH = 'preprocessor.joblib'

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREP_PATH):
    print('WARNING: model.joblib or preprocessor.joblib not found. Run train.py first.')
    model = None
    preprocessor = None
else:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREP_PATH)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return 'Model not found. Run training first.', 400
    data = request.form.to_dict()
    # expected keys: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
    df = pd.DataFrame([data])
    # convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]
    label = int(prob > 0.5)
    return render_template('result.html', prob=round(float(prob),4), label=label)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None or preprocessor is None:
        return jsonify({'error':'Model not trained'}), 400
    payload = request.get_json()
    df = pd.DataFrame([payload])
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]
    label = int(prob > 0.5)
    return jsonify({'probability': float(prob), 'isFraud': label})

if __name__ == '__main__':
    app.run(debug=True)

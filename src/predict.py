import joblib
import pandas as pd

# Load model & encoders
model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")

def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    return df

def predict(data_dict):
    df = preprocess_input(data_dict)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob > 0.5)

    return {
        "fraud": pred,
        "probability": float(prob)
    }
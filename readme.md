# Fraud Detection System

A machine learning-based fraud detection system with real-time API deployment using FastAPI. Implemented advanced models including XGBoost and LightGBM for high-performance fraud detection

## Features
- Data preprocessing & feature engineering
- Handles class imbalance
- Trained ML models (Random Forest)
- REST API for real-time fraud prediction
- Modular and scalable architecture

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- FastAPI
- Joblib

## Workflow
1. Data preprocessing
2. Model training
3. Evaluation
4. Deployment via API

## Run Locally

```bash
pip install -r requirements.txt
python src/train.py
uvicorn api.app:app --reload

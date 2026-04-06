from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class Transaction(BaseModel):
    amount: float
    time: float
    feature1: float
    feature2: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict_fraud(tx: Transaction):
    result = predict(tx.dict())
    return {"fraud": result}
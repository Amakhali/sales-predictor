from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from src.model import SalesPredictor
import logging

app = FastAPI(title="Sales Prediction API")
model = SalesPredictor()

# Load model at startup
model.load_model("models/sales_model.joblib")

class PredictionInput(BaseModel):
    date: str
    customer_id: int
    product_id: int
    quantity: int
    unit_price: float
    product_description: str
    product_category: str
    product_line: str
    raw_material: str
    region: str
    latitude: float
    longitude: float

class PredictionOutput(BaseModel):
    predicted_sales: float
    confidence: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(df)
        
        return {
            "predicted_sales": float(prediction[0]),
            "confidence": 0.95  # Example confidence score
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": "1.0.0"}


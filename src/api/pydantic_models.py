from pydantic import BaseModel
from typing import List

# Request model
class CustomerData(BaseModel):
    recency_days: float
    frequency: float
    monetary: float
    cluster: int

# Response model
class PredictionResponse(BaseModel):
    customer_id: int
    risk_probability: float
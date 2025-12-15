from fastapi import FastAPI
import mlflow.pyfunc
from src.api.pydantic_models import CustomerData, PredictionResponse

app = FastAPI(title="Credit Risk Prediction API")

# Load best model from MLflow
model = mlflow.pyfunc.load_model("models:/CreditRiskModel/Production")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    input_df = data.dict()
    risk_prob = model.predict([input_df])[0]  # returns probability
    return PredictionResponse(customer_id=123, risk_probability=risk_prob)
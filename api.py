from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import requests
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Load Open-Source LLM (Falcon)
llm_pipeline = pipeline("text-generation", model="facebook/opt-1.3b") 

# Initialize FastAPI
app = FastAPI()

# Verify API Key
def verify_api_key(request: Request):

    api_key = request.headers.get('x-api-key')

    if api_key is None:
        raise HTTPException(status_code=400, detail="API Key missing in headers")

    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return api_key

# Load models
with open("credit_risk_model.pkl", "rb") as f:
    credit_risk_model = pickle.load(f)
with open("transaction_categorization_model.pkl", "rb") as f:
    transaction_model = pickle.load(f)

# Request Models
class CreditRiskRequest(BaseModel):
    credit_score: int
    debt_to_income_ratio: float
    missed_payments: int
    loan_utilization: float
    total_outstanding_debt: float

class TransactionRequest(BaseModel):
    description: str

######################################### API Endpoints ############################################
@app.get("/")
async def home(request: Request):

    return {"message": "Loaded correctly home page"}


@app.post("/predict-credit-risk")
async def predict_credit_risk(
    request: Request, 
    api_key: str = Depends(verify_api_key)
):

     # Extract JSON input from request
    data = await request.json()

    # Ensure required fields exist
    required_fields = ["credit_score", "missed_payments", "total_outstanding_debt", "debt_to_income_ratio"]
    for field in required_fields:
        if field not in data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    # Prepare input data for model
    input_data = {
    "Credit_Score": data["credit_score"],                            # Anywhere between 300 - 900 
    "Debt_to_Income_Ratio": data["debt_to_income_ratio"],            # 0 - 4
    "Missed_Payments": data["missed_payments"],                      # 0 - 12
    "Total_Debt": data["total_outstanding_debt"],                    # 40,000 - 5,00,000
    "Existing_Loans": data["credit_score"]                           # 0 - 12
    }
    
    df = pd.DataFrame([input_data])

    print("📌 Model Feature Names:", credit_risk_model.feature_names_in_)

    if credit_risk_model:
        prediction = credit_risk_model.predict(df)
    else:
        prediction = ["No model loaded!"]

    return {"risk_level": prediction[0]}

@app.post("/categorize-transaction")
def categorize_transaction(
    data: TransactionRequest, api_key: str = Depends(verify_api_key)
):
    prediction = transaction_model.predict([data.description])
    return {"category": prediction[0]}

@app.post("/generate-financial-insights")
async def generate_insights(    
    request: Request, 
    api_key: str = Depends(verify_api_key)
):
    """Generate AI-based financial insights using an LLM"""

    # Extract JSON input from request
    data = await request.json()

    # Ensure required fields exist
    required_fields = ["credit_score", "missed_payments", "total_outstanding_debt", "debt_to_income_ratio"]
    for field in required_fields:
        if field not in data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    # Prepare input data for model
    input_data = {
    "Credit_Score": data["credit_score"],                            # Anywhere between 300 - 900 
    "Debt_to_Income_Ratio": data["debt_to_income_ratio"],            # 0 - 4
    "Missed_Payments": data["missed_payments"],                      # 0 - 12
    "Total_Debt": data["total_outstanding_debt"],                    # 40,000 - 5,00,000
    "Existing_Loans": data["credit_score"]                           # 0 - 12
    }
    
    # Convert input data into DataFrame for prediction
    df = pd.DataFrame([input_data])

    # Get prediction from the model
    prediction = credit_risk_model.predict(df)

    # Create financial insight prompt
    prompt = f"""
    The user has the following financial details: \n
    Credit Score: {data["credit_score"]} \n
    Debt-to-Income Ratio: {data["debt_to_income_ratio"]} \n
    Missed Payments: {data["missed_payments"]} \n
    Total Outstanding Debt: {data["total_outstanding_debt"]} \n\n

    Based on the provided information, the model predicts the following credit risk level: {prediction[0]}.\n 

    avoid unnecessary information.
    """

    # Generate response from LLM
    response = llm_pipeline(prompt, max_length=None, max_new_tokens=100, do_sample=True)[0]['generated_text']

    return {"financial_insights": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
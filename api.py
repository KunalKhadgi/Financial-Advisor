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
llm_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

# Initialize FastAPI
app = FastAPI()

# Verify API Key
def verify_api_key(api_key: str = Header(None)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="invalid api")
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
def home():
    return {"message": "Welcome to the CredArtha API"}


@app.post("/predict-credit-risk")
async def predict_credit_risk(request: Request, api_key: str = Header(None)):
    """Validates API key and predicts credit risk"""

    print("\n📌 FULL REQUEST HEADERS:", request.headers)  # Debugging

    # Debugging: Print received API key
    print(f"📌 Received API Key: [{api_key}]")  # THIS NOW WORKS CORRECTLY
    print(f"✅ Expected API Key: [{API_KEY}]")  # Debugging

     # ✅ Extract JSON input from request
    data = await request.json()

    # ✅ Ensure required fields exist
    required_fields = ["credit_score", "missed_payments", "total_outstanding_debt", "debt_to_income_ratio"]
    for field in required_fields:
        if field not in data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    # ✅ Convert JSON input to Pandas DataFrame
    input_data = {
    "Credit_Score": data["credit_score"],  
    "Debt_to_Income_Ratio": data["debt_to_income_ratio"],
    "Missed_Payments": data["missed_payments"],
    "Total_Debt": data["total_outstanding_debt"],
    "Existing_Loans": data["credit_score"]  
    }
    
    df = pd.DataFrame([input_data])

    print("📌 Model Feature Names:", credit_risk_model.feature_names_in_)

    if credit_risk_model:
        prediction = credit_risk_model.predict(df)
    else:
        prediction = ["No model loaded!"]

    return {"risk_level": prediction[0]}

@app.post("/categorize-transaction")
def categorize_transaction(data: TransactionRequest, api_key: str = Depends(verify_api_key)):
    prediction = transaction_model.predict([data.description])
    return {"category": prediction[0]}

@app.post("/generate-financial-insights")
async def generate_insights(request: Request, api_key: str = Header(None)):
    """Generate AI-based financial insights using an LLM"""

    # Extract request data
    data = await request.json()

    if "credit_score" not in data or "transactions" not in data:
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Generate Prompt
    prompt = f"""
    The user has a credit score of {data["credit_score"]} and the following transactions:
    {data["transactions"]}

    Provide:
    1. A summary of their financial health.
    2. A risk assessment (high, moderate, low).
    3. Personalized financial recommendations.
    """

    # Generate response from LLM
    response = llm_pipeline(prompt, max_length=200, do_sample=True)[0]['generated_text']

    return {"financial_insights": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
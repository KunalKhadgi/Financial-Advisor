Overview
This API provides functionalities for credit risk assessment, transaction categorization, and AI-based financial insights. It leverages FastAPI for API development, machine learning models for financial predictions, and a transformer-based language model for financial insights.

**FEATURE**
_1. Credit Risk Prediction:_

Uses a Random Forest model to predict a user's credit risk based on their financial data.

**FEATURE**_2. Transaction Categorization:_

   Uses an NLP model to classify financial transactions into predefined categories.

**FEATURE**_3. AI-Based Financial Insights:_

   _Utilizes the **facebook/opt-1.3b** LLM to generate personalized financial insights and recommendations._

Project Structure
```
├── api.py                        # FastAPI-based API implementation
├── data.py                       # Synthetic data generation for transactions and credit reports
├── models.py                     # ML models for credit risk and transaction categorization
├── synthetic_transactions.csv     # Generated transactions data
├── synthetic_credit_report.csv    # Generated credit report data
├── transaction_categorization_model.pkl  # Saved transaction categorization model
├── credit_risk_model.pkl          # Saved credit risk model
├── requirements.txt              # List of project dependencies
├── .env                          # Environment file for sensitive API keys
```
**Installation
Prerequisites**

Python 3.8+: Ensure Python 3.8 or higher is installed.

_Install Dependencies_
Run the following command to install all the required dependencies:
```
pip install -r requirements.txt
```

**Usage
Running the API**
To start the FastAPI application, use the following command:

```
uvicorn api:app --host 127.0.0.1 --port 8080 --reload
```

This will start the API locally on http://127.0.0.1:8080.

_API Endpoints_

* ```Home (GET /):```

Returns a simple welcome message to confirm the API is running.

* ```Predict Credit Risk (POST /predict-credit-risk):```

Input: JSON data with financial fields.

Example Request:
```
{
  "credit_score": 500,
  "debt_to_income_ratio": 0.35,
  "missed_payments": 2,
  "loan_utilization": 0.4,
  "total_outstanding_debt": 15000
}
```
Output: Predicted credit risk category (e.g., "High", "Moderate", "Low").

* ```Categorize Transaction (POST /categorize-transaction):```

Input: A JSON body with a transaction description.

Example Request:
```
{
  "description": "Payment for electricity bill"
}
```
Output: Predicted category for the transaction (e.g., "utilities", "loan").

# Generate Financial Insights (POST /generate-financial-insights):

Input: JSON data with financial details (credit score, debt-to-income ratio, missed payments, etc.).

Example Request:
```
{
  "credit_score": 500,
  "debt_to_income_ratio": 0.35,
  "missed_payments": 2,
  "total_outstanding_debt": 15000
}
```
Output: Personalized financial insights and recommendations.

# Environment Variables
 To keep your API key secure, create a .env file and set your API key in it:
 API_KEY = your_api_key
 

License
This project is licensed under the MIT License.


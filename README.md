# Financial-Advisor
Advises client based on his/her finance data

API provides functionalities for credit risk assessment, transaction categorization, and AI-based financial insights. It leverages FastAPI for API development, machine learning models for financial predictions, and a transformer-based language model for financial insights.

Features

Credit Risk Prediction: Uses a Random Forest model to predict a user's credit risk based on financial data.

Transaction Categorization: Uses an NLP model to classify financial transactions into predefined categories.

AI-Based Financial Insights: Utilizes a Falcon-7B LLM to generate personalized financial insights.

Project Structure

├── api.py                # FastAPI-based API implementation

├── data.py               # Synthetic data generation for transactions and credit reports

├── models.py             # ML models for credit risk and transaction categorization

├── synthetic_transactions.csv  # Generated transactions data

├── synthetic_credit_report.csv # Generated credit report data

├── transaction_categorization_model.pkl  # Saved transaction model

├── credit_risk_model.pkl  # Saved credit risk model

**Installation**

  Prerequisites

  Python 3.8+

  pip

**Install Dependencies**

  pip install -r requirements.txt

Usage

**Running the API**

  uvicorn api:app --reload

API will be accessible at http://127.0.0.1:8080

**API Endpoints**

  *Home (GET /):*

    Returns a welcome message.

  *Predict Credit Risk (POST /predict-credit-risk):*

    Expects JSON input with credit-related fields.

    Returns predicted credit risk category.

  *Categorize Transaction (POST /categorize-transaction):*

    Takes a transaction description and returns the predicted category.

  *Generate Financial Insights (POST /generate-financial-insights):*

    Uses LLM to generate personalized financial recommendations.

*Environment Variables*

Create a .env file and set your API key:

  API_KEY=your_secure_api_key

**License**

  MIT License

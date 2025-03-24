import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
from sqlalchemy import create_engine
import pickle

syn_cre_rep = pd.read_csv('synthetic_credit_report.csv')
syn_tran = pd.read_csv('synthetic_transactions.csv')

syn_cre_rep.info()
syn_tran.info()

# Data Cleaning Function
def clean_transactions(df):
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Transaction_Type"] = df["Transaction_Type"].str.lower().str.strip()
    df["Description"] = df["Description"].str.lower().str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
    df.dropna(inplace=True)
    return df

# Clean Data
syn_tran = clean_transactions(syn_tran)
syn_cre_rep.dropna(inplace=True)

# Database Connection
engine = create_engine('sqlite:///financial_data.db')
syn_tran.to_sql('transactions', engine, if_exists='replace', index=False)
syn_cre_rep.to_sql('credit_reports', engine, if_exists='replace', index=False)

print("ETL Pipeline executed: Data cleaned and loaded into database successfully!")

#########################################################################################

# Load Data
syn_tran = pd.read_csv('synthetic_transactions.csv')

# Data Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

syn_tran["Cleaned_Description"] = syn_tran["Description"].astype(str).apply(clean_text)

# Define categories
categories = {
    "salary": ["salary", "credit"],
    "shopping": ["amazon", "flipkart", "myntra"],
    "food": ["swiggy", "zomato", "starbucks"],
    "loan": ["loan", "emi", "payment"],
    "utilities": ["electricity", "water", "broadband", "gas"],
    "cash": ["atm", "withdrawal"],
    "transfer": ["upi", "imps", "neft"]
}

# Label Transactions
def categorize_transaction(description):
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "other"

syn_tran["Category"] = syn_tran["Cleaned_Description"].apply(categorize_transaction)

# Train NLP Model
X = syn_tran["Cleaned_Description"]
y = syn_tran["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

transaction_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
transaction_model.fit(X_train, y_train)

# Save Transaction Categorization Model immediately after training
with open("transaction_categorization_model.pkl", "wb") as f:
    pickle.dump(transaction_model, f)
print("Transaction Categorization Model saved as transaction_categorization_model.pkl")

# Evaluate NLP Model
y_pred = transaction_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Transaction Categorization Model Accuracy: {accuracy:.2f}")

# Save categorized transactions
syn_tran.to_csv("categorized_transactions.csv", index=False)

print("Transactions categorized using NLP!") 

###########################################################################################

# Load Credit Bureau Data
file_path = "synthetic_credit_report.csv"
syn_cre_rep = pd.read_csv(file_path)

# Define Risk Category based on Credit Score
bins = [300, 600, 750, 900]
labels = ['High Risk', 'Medium Risk', 'Low Risk']
syn_cre_rep['Risk_Category'] = pd.cut(syn_cre_rep['Credit_Score'], bins=bins, labels=labels)

# Define Features and Target
features = ["Credit_Score", "Debt_to_Income_Ratio", "Missed_Payments", "Total_Debt", "Existing_Loans"]
target = "Risk_Category"

# Prepare Data
X = syn_cre_rep[features]
y = syn_cre_rep[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Credit Risk Model
credit_risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
credit_risk_model.fit(X_train, y_train)

# Save Credit Risk Model immediately after training
with open("credit_risk_model.pkl", "wb") as file:
    pickle.dump(credit_risk_model, file)
print("Credit Risk Model saved as credit_risk_model.pkl")

# Evaluate Credit Risk Model
y_pred = credit_risk_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Credit Risk Model Accuracy using Random forest: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Explain Model with SHAP
explainer = shap.Explainer(credit_risk_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

print("Credit risk assessment model trained and analyzed!")

import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
random.seed(42)

# Define categories
categories = {
    "Salary Credits": ["Salary Credit - Infosys Ltd", "Salary Credit - TCS Ltd", "Salary Credit - Wipro Ltd"],
    "Shopping": ["Amazon Purchase", "Flipkart Purchase", "Myntra Shopping"],
    "Food & Dining": ["Swiggy Order", "Zomato Payment", "Starbucks"],
    "Loan EMI Payments": ["HDFC Loan EMI Payment", "SBI Home Loan EMI", "ICICI EMI Payment"],
    "Utility Bills": ["Electricity Bill", "Water Bill", "Broadband Payment", "Gas Bill"],
    "Cash Withdrawals": ["ATM Withdrawal", "Bank Cash Withdrawal"],
    "Transfers": ["UPI Transfer", "IMPS Transfer", "NEFT Transfer"]
}

# Generate synthetic financial transactions
def generate_transactions(customer_id, num_transactions=60):
    transactions = []
    for _ in range(num_transactions):
        date = fake.date_between(start_date='-1y', end_date='today')
        description = random.choice(random.choice(list(categories.values())))
        amount = round(random.uniform(100, 50000), 2)
        transaction_type = "Credit" if "Salary" in description else "Debit"
        transactions.append([customer_id, date, description, amount, transaction_type])
    return transactions

# Generate credit bureau report
def generate_credit_report(customer_id):
    credit_score = random.randint(300, 900)
    existing_loans = random.randint(0, 5)
    missed_payments = random.randint(0, 12)
    total_debt = round(random.uniform(5000, 500000), 2)
    income = round(random.uniform(30000, 200000), 2)
    debt_to_income_ratio = round(total_debt / income, 2)
    return [customer_id, credit_score, existing_loans, missed_payments, total_debt, debt_to_income_ratio]

# Generate data for multiple customers
num_customers = 100
all_transactions = []
all_credit_reports = []

for customer_id in range(1, num_customers + 1):
    all_transactions.extend(generate_transactions(customer_id))
    all_credit_reports.append(generate_credit_report(customer_id))

# Convert to DataFrame
transactions_df = pd.DataFrame(all_transactions, columns=["Customer_ID", "Date", "Description", "Amount", "Transaction_Type"])
credit_report_df = pd.DataFrame(all_credit_reports, columns=["Customer_ID", "Credit_Score", "Existing_Loans", "Missed_Payments", "Total_Debt", "Debt_to_Income_Ratio"])

# Introduce inconsistencies
for _ in range(5):
    transactions_df.at[random.randint(0, len(transactions_df)-1), "Date"] = random.choice([
        fake.date(), fake.date_object(), fake.date_of_birth(minimum_age=18, maximum_age=65)
    ])
    transactions_df.at[random.randint(0, len(transactions_df)-1), "Amount"] = None
    transactions_df.at[random.randint(0, len(transactions_df)-1), "Transaction_Type"] = random.choice(["credit", "debit"])

# Save to CSV
transactions_df.to_csv("synthetic_transactions.csv", index=False)
credit_report_df.to_csv("synthetic_credit_report.csv", index=False)

print("Synthetic financial transaction and credit report datasets generated successfully!")
# Install if needed:
# pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Load the mock data from the CSV file
try:
    df = pd.read_csv('/content/creditcarddata.csv')
    print("Mock data loaded successfully.")
    print("First 5 rows of the data:")
    print(df.head())
except FileNotFoundError:
    print("Error: MOCK_DATA (1).csv not found. Please ensure the file is in the correct directory.")
    # If the file is not found, we'll use a hardcoded dataset for demonstration
    data = {
        "Annual_Income":        [8000, 7500, 9000, 3000, 3500, 4000, 8500, 2000, 2500, 7000, 7200, 2800],
        "Debt_to_Income":       [0.2, 0.25, 0.18, 0.6, 0.55, 0.5, 0.22, 0.7, 0.65, 0.3, 0.28, 0.68],
        "Credit_Score":         [780, 750, 800, 500, 520, 550, 770, 480, 510, 720, 730, 490],
        "Missed_Payments":      [0, 1, 0, 5, 4, 3, 1, 6, 5, 2, 1, 6],
        "Loan_Default_History": [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1],
        "Credit_History_Years": [10, 8, 12, 3, 4, 5, 9, 2, 3, 7, 8, 2]
    }
    df = pd.DataFrame(data)
    print("Using hardcoded data for demonstration.")
    print("First 5 rows of the data:")
    print(df.head())

# Select relevant numerical columns for Factor Analysis
# Convert 'Loan_Default_History' to integer (0 or 1) if it's boolean
if 'Loan_Default_History' in df.columns:
    df['Loan_Default_History'] = df['Loan_Default_History'].astype(int)

# Prompt user to define numerical columns for factor analysis
user_input_columns = input("Enter comma-separated column names for Factor Analysis (e.g., credit_limit,credit_score,annual_income). Press Enter for default columns: ")

default_analysis_columns = [
    'credit_limit', 'credit_score', 'annual_income', 'loan_amount', 
    'loan_term', 'interest_rate', 'Debt_to_Income', 'Missed_Payments', 
    'Loan_Default_History', 'Credit_History_Years'
]

if user_input_columns.strip():
    requested_columns = [col.strip() for col in user_input_columns.split(',')]
    # Filter for columns that actually exist in the DataFrame
    analysis_columns = [col for col in requested_columns if col in df.columns]
    if not analysis_columns:
        print("Warning: None of the specified columns were found in the data. Using default columns.")
        analysis_columns = default_analysis_columns
    else:
        print(f"Using specified columns for analysis: {analysis_columns}")
else:
    # Default columns if user input is empty
    print("Using default columns for Factor Analysis.")
    analysis_columns = default_analysis_columns

# Filter DataFrame to include only these columns
# Also handle any remaining non-numeric data or NaNs by dropping rows or imputation if necessary
pd_df = df[analysis_columns].copy()
pd_df = pd_df.apply(pd.to_numeric, errors='coerce').dropna()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pd_df)

# Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
fa.fit(X_scaled)

# Factor Loadings
loadings = pd.DataFrame(
    fa.components_.T,
    index=pd_df.columns,
    columns=["Factor1", "Factor2"]
)

print("\nFactor Loadings:")
print(loadings.round(3))

# Factor Scores
factor_scores = fa.transform(X_scaled)
factor_scores_df = pd.DataFrame(
    factor_scores,
    columns=["Factor1_Score", "Factor2_Score"]
)

# -------------------------------
# 📊 Visualization 1: Factor Loadings Bar Chart
# -------------------------------
loadings.plot(kind='bar')
plt.title("Factor Loadings (Credit Risk Drivers)")
plt.xlabel("Variables")
plt.ylabel("Loading Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 📊 Visualization 2: Borrower Risk Distribution (Scatter Plot)
# -------------------------------
plt.scatter(
    factor_scores_df["Factor1_Score"],
    factor_scores_df["Factor2_Score"]
)

# Annotate each borrower
for i in range(len(factor_scores_df)):
    plt.text(
        factor_scores_df["Factor1_Score"][i],
        factor_scores_df["Factor2_Score"][i],
        str(i)
    )

plt.title("Borrower Risk Segmentation")
plt.xlabel("Factor 1 (Repayment Capacity)")
plt.ylabel("Factor 2 (Credit Behaviour Risk)")
plt.grid()
plt.show()
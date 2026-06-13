import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
# df.drop("Loan_ID", axis=1, inplace=True)
df.head()
# handle missing values
df.isnull().sum()
# fill missing values
# 1️⃣ Drop Loan_ID
df.drop("Loan_ID", axis=1, inplace=True)

# 2️⃣ Fix Dependents text value
df["Dependents"] = df["Dependents"].replace("3+", 3)

# 3️⃣ Fill missing value in Dependents
df["Dependents"] = df["Dependents"].fillna(0)

# 4️⃣ Now convert to integer
df["Dependents"] = df["Dependents"].astype(int)

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

# Numerical columns → median
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())

df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].median())
# Encode Categorical Variables
le = LabelEncoder()

categorical_cols = [
    "Gender", "Married", "Education",
    "Self_Employed", "Property_Area", "Loan_Status"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    # Feature Scaling
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train‑Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# ML part start
# Train the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Make Predictions
y_pred = model.predict(X_test)
# Model Evaluation 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
import streamlit as st

st.set_page_config(page_title="Loan Approval App", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details and click Predict")

# ---- User Inputs ----
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
app_income = st.number_input("Applicant Income", min_value=0)
co_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---- Convert input to model format ----
input_data = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": co_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area
}

input_df = pd.DataFrame([input_data])

# SAME preprocessing apply karo
input_df = pd.get_dummies(input_df)

# Columns align karo (VERY IMPORTANT)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_df)

# ---- Prediction ----
if st.button("🔮 Predict Loan Status"):
    result = model.predict(input_scaled)[0]
    if result == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

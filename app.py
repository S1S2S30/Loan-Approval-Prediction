# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

 
# PAGE CONFIG
 
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")

 
# LOAD DATA
 
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

 
# SIDEBAR NAVIGATION
 
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Introduction", "EDA", "Model & Prediction", "Conclusion"]
)

 
# PREPROCESSING (RAW DATA)
 
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Loan_Status_Y", axis=1)
y = df_encoded["Loan_Status_Y"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

 
# MODEL TRAINING
 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

 
# INTRODUCTION SECTION
 
if section == "Introduction":
    st.header("📌 Introduction")
    st.write("""
    This project predicts whether a loan application will be approved or not
    using machine learning techniques.

    A raw loan approval dataset was used containing demographic and financial
    information of applicants. The project demonstrates a complete data science
    workflow including EDA, preprocessing, model training, and deployment using Streamlit.
    """)

 
# EDA SECTION
 
elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Loan Status Distribution")
    st.bar_chart(df["Loan_Status"].value_counts())

    st.write("""
    **Key Insights:**
    - Applicants with credit history have higher approval chances  
    - Missing values were present and handled during preprocessing  
    - Loan approval depends on multiple financial and demographic factors
    """)

 
# MODEL & PREDICTION SECTION
 
elif section == "Model & Prediction":
    st.header("🤖 Model & Prediction")

    st.write(f"**Model Used:** Logistic Regression")
    st.write(f"**Model Accuracy:** {accuracy:.2f}")

    st.subheader("Enter Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", 0)

    with col2:
        coapplicant_income = st.number_input("Coapplicant Income", 0)
        loan_amount = st.number_input("Loan Amount", 0)
        loan_term = st.number_input("Loan Amount Term", 0)
        credit_history = st.selectbox("Credit History", [1.0, 0.0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict Loan Status"):
        input_data = {
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Dependents": dependents,
            "Gender_Male": 1 if gender == "Male" else 0,
            "Married_Yes": 1 if married == "Yes" else 0,
            "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
            "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
            "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
            "Property_Area_Urban": 1 if property_area == "Urban" else 0
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

 
# CONCLUSION SECTION
 
elif section == "Conclusion":
    st.header("✅ Conclusion")

    st.write("""
    This project successfully demonstrates an end-to-end data science pipeline.
    Starting from raw data, preprocessing and exploratory analysis were performed,
    followed by machine learning model training and deployment using Streamlit.

    The application allows real-time loan approval prediction based on user input,
    making the system interactive and user-friendly.
    """)

st.markdown("---")
st.caption("Semester Project | Tools & Techniques in Data Science")

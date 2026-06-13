# 🏦 Loan Approval Prediction

A Machine Learning model that predicts whether a loan application will be **Approved** or **Rejected**, based on applicant details such as income, credit history, education, and employment status.

🔗 **Live Demo:** [Try it on HuggingFace Spaces](https://huggingface.co/spaces/S-a-r-a/Loan-Approval-Prediction)

---

## 📌 Project Overview

Banks receive thousands of loan applications and manually reviewing each one is slow and inconsistent. This project builds a **Logistic Regression** model that automates the initial screening — predicting loan approval likelihood based on historical applicant data.

---

## 📊 Dataset

| Detail | Value |
|---|---|
| Total Records | 614 |
| Features | 12 |
| Target Variable | `Loan_Status` (Y / N) |
| Class Distribution | 422 Approved, 192 Rejected |

### Key Features Used
- Gender, Marital Status, Dependents
- Education, Self-Employed
- Applicant & Co-applicant Income
- Loan Amount & Loan Term
- Credit History
- Property Area

---

## 🧹 Data Preprocessing

- Handled missing values using **mode** (categorical) and **median** (numerical) imputation
- Converted `Dependents` ("3+") to numeric format
- One-hot encoding for categorical variables
- Feature scaling using **StandardScaler**

---

## 🤖 Model

| Detail | Value |
|---|---|
| Algorithm | Logistic Regression |
| Train/Test Split | 80/20 |
| Library | scikit-learn |

---

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — Data handling
- **Scikit-learn** — ML model & preprocessing
- **Streamlit** — Web app interface

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/msdsf25a001-dotcom/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
loan-approval-prediction/
├── app.py                          # Streamlit web app
├── train_u6lujuX_CVtuZ9i.csv       # Training dataset
├── test_Y3wMUE5_7gLdaTN.csv        # Test dataset
├── notebooks/                      # EDA & preprocessing notebooks
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- Try ensemble models (Random Forest, XGBoost) for better accuracy
- Add SHAP/LIME for model explainability
- Handle class imbalance with SMOTE
- Deploy with FastAPI for production-grade API

---

## 👩‍💻 Author

**Sara Saeed** — MPhil Data Science, PUCIT, Lahore

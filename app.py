
import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏦 Loan Approval Prediction App")

# Sidebar form
st.sidebar.header("Applicant Details")

# Collect user inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.sidebar.selectbox("Loan Term (in days)", [360, 180, 240, 300, 120])
credit_history = st.sidebar.selectbox("Credit History", [1, 0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode categorical values as in training
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Predict button
if st.sidebar.button("Predict Loan Status"):
    # Form input vector
    input_data = np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_term, credit_history, property_area]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    st.subheader("Prediction Result:")
    st.success(result if prediction == 1 else "")
    st.error(result if prediction == 0 else "")

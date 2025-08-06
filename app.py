import streamlit as st
import joblib
import numpy as np
import pandas as pd 

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
loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0)
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

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import base64

# ----- Custom Background -----
def set_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1620138548241-dfeee58b1fbf"); 
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_url()

# ----- Charts Section -----
st.header("📊 Applicant Financial Overview")

# Chart 1: Bar chart - Applicant vs Coapplicant Income
fig1, ax1 = plt.subplots()
sns.barplot(x=["Applicant", "Coapplicant"], y=[applicant_income, coapplicant_income], palette="Blues_d", ax=ax1)
ax1.set_ylabel("Income (₹)")
ax1.set_title("Applicant vs Coapplicant Income")
st.pyplot(fig1)

# Chart 2: Scatter plot - Income vs Loan Amount
fig2, ax2 = plt.subplots()
total_income = applicant_income + coapplicant_income
ax2.scatter(total_income, loan_amount, color='green')
ax2.set_xlabel("Total Income (₹)")
ax2.set_ylabel("Loan Amount (₹)")
ax2.set_title("Total Income vs Loan Amount")
st.pyplot(fig2)

# ----- PDF Generation -----
def generate_pdf(prediction_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("🏦 Final Loan Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Prediction Result: {prediction_text}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Applicant Income: ₹{applicant_income}", styles["Normal"]))
    elements.append(Paragraph(f"Coapplicant Income: ₹{coapplicant_income}", styles["Normal"]))
    elements.append(Paragraph(f"Loan Amount: ₹{loan_amount}", styles["Normal"]))
    elements.append(Paragraph(f"Loan Term: {loan_term} days", styles["Normal"]))
    elements.append(Paragraph(f"Credit History: {credit_history}", styles["Normal"]))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

if 'prediction' in locals():
    pdf_buffer = generate_pdf(result)
    b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')
    st.download_button(
        label="📄 Download Final Loan Prediction Report (PDF)",
        data=base64.b64decode(b64_pdf),
        file_name="Loan_Prediction_Report.pdf",
        mime="application/pdf"
    )





import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Sample training-like dataset for visualization
sample_data = pd.DataFrame({
    'Loan_Status': [1, 0, 1, 1, 0, 1, 0],
    'LoanAmount': [114, 128, 66, 120, 150, 110, 135],
    'ApplicantIncome': [5849, 4583, 3000, 2583, 4200, 6100, 3900],
    'CoapplicantIncome': [0, 1508, 0, 2358, 0, 1300, 1800]
})

st.title("🏦 Loan Approval Prediction App")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Term", [360, 180, 300])
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

input_data = np.array([[
    gender, married, dependents, education, self_employed,
    applicant_income, coapplicant_income, loan_amount,
    loan_term, credit_history, property_area
]])

# Predict
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result_text = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.success(result_text if prediction == 1 else "")
    st.error(result_text if prediction == 0 else "")

    # Append to sample data
    sample_data.loc[len(sample_data.index)] = [
        prediction, loan_amount, applicant_income, coapplicant_income
    ]

    # 1️⃣ Loan Status Bar Chart
    st.subheader("📊 Loan Status Count")
    st.bar_chart(sample_data['Loan_Status'].value_counts())

    # 2️⃣ Loan Amount Histogram
    st.subheader("📉 Loan Amount Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(sample_data['LoanAmount'], bins=10, color='skyblue', edgecolor='black')
    ax1.set_xlabel("Loan Amount")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    # 3️⃣ Box Plot: Income by Loan Status
    st.subheader("📦 Applicant Income by Loan Status")
    fig2, ax2 = plt.subplots()
    sample_data.boxplot(column='ApplicantIncome', by='Loan_Status', ax=ax2)
    ax2.set_title("Income Distribution")
    st.pyplot(fig2)

    # 4️⃣ New Chart: Applicant vs Coapplicant Income
    st.subheader("📍 Applicant vs Coapplicant Income")
    fig3, ax3 = plt.subplots()
    ax3.scatter(sample_data['ApplicantIncome'], sample_data['CoapplicantIncome'], color='green')
    ax3.set_xlabel("Applicant Income")
    ax3.set_ylabel("Coapplicant Income")
    st.pyplot(fig3)

    # 5️⃣ New Chart: Income vs Loan Bar
    st.subheader("📌 Income vs Loan Amount")
    fig4, ax4 = plt.subplots()
    categories = ['Total Income', 'Loan Amount']
    values = [applicant_income + coapplicant_income, loan_amount]
    ax4.bar(categories, values, color=['orange', 'blue'])
    st.pyplot(fig4)

    # 6️⃣ PDF Report Generation
    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("📄 Loan Prediction Report", styles['Title']),
            Paragraph(f"Result: {result_text}", styles['Normal']),
            Paragraph(f"Applicant Income: ₹{applicant_income}", styles['Normal']),
            Paragraph(f"Coapplicant Income: ₹{coapplicant_income}", styles['Normal']),
            Paragraph(f"Total Income: ₹{applicant_income + coapplicant_income}", styles['Normal']),
            Paragraph(f"Loan Amount: ₹{loan_amount}", styles['Normal']),
            Paragraph(f"Loan Term: {loan_term} months", styles['Normal']),
            Paragraph(f"Credit History: {credit_history}", styles['Normal']),
        ]
        doc.build(story)
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf()
    st.download_button("📥 Download Loan Report PDF", data=pdf_buffer, file_name="loan_report.pdf", mime="application/pdf")

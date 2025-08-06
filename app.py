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
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 1. Bar Chart: Applicant, Coapplicant, Loan
st.subheader("💹 Income vs Loan Amount")

loan_amount_actual = loan_amount * 100
categories = ['Applicant Income', 'Coapplicant Income', 'Loan Amount']
values = [applicant_income, coapplicant_income, loan_amount_actual]

fig1, ax1 = plt.subplots()
bars = ax1.bar(categories, values, color=['skyblue', 'lightgreen', 'salmon'])
ax1.set_ylabel("Amount (₹)")
ax1.set_title("Income vs Loan Amount")
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(bar.get_height())}',
             ha='center', va='bottom')
st.pyplot(fig1)

# 2. Pie Chart: Applicant vs Coapplicant Income
st.subheader("📈 Income Contribution Pie Chart")
income_parts = [applicant_income, coapplicant_income]
labels = ['Applicant', 'Coapplicant']
fig2, ax2 = plt.subplots()
ax2.pie(income_parts, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
ax2.axis('equal')
st.pyplot(fig2)

# 3. Horizontal Bar Chart: Contribution
st.subheader("📊 Income Contribution Comparison")
fig3, ax3 = plt.subplots()
ax3.barh(['Applicant', 'Coapplicant'], [applicant_income, coapplicant_income], color=['skyblue', 'lightgreen'])
ax3.set_xlabel("Income Amount (₹)")
st.pyplot(fig3)

# 4. Downloadable PDF Report
st.subheader("📄 Download Loan Prediction Report")

def generate_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)

    text.textLine("Loan Prediction Report")
    text.textLine("----------------------------")
    text.textLine(f"Gender: {'Male' if gender == 1 else 'Female'}")
    text.textLine(f"Married: {'Yes' if married == 1 else 'No'}")
    text.textLine(f"Dependents: {dependents}")
    text.textLine(f"Education: {'Graduate' if education == 1 else 'Not Graduate'}")
    text.textLine(f"Self Employed: {'Yes' if self_employed == 1 else 'No'}")
    text.textLine(f"Applicant Income: ₹{applicant_income}")
    text.textLine(f"Coapplicant Income: ₹{coapplicant_income}")
    text.textLine(f"Loan Amount: ₹{loan_amount_actual}")
    text.textLine(f"Loan Term: {loan_term} days")
    text.textLine(f"Credit History: {'Good' if credit_history == 1 else 'Poor'}")
    text.textLine(f"Property Area: {['Rural', 'Semiurban', 'Urban'][property_area]}")
    text.textLine("----------------------------")
    text.textLine(f"Prediction: {result}")
    
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

pdf_buffer = generate_pdf()

st.download_button(
    label="📥 Download PDF Report",
    data=pdf_buffer,
    file_name="loan_prediction_report.pdf",
    mime="application/pdf"
)


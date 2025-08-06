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

# -----------------------------
# Chart 1: Applicant vs Coapplicant Income
# -----------------------------
st.subheader("📊 Applicant vs Coapplicant Income")
fig1, ax1 = plt.subplots()
bars1 = ax1.bar(['Applicant', 'Coapplicant'], [applicant_income, coapplicant_income], color=['skyblue', 'lightgreen'])
ax1.set_ylabel("Income (₹)")
ax1.set_title("Applicant vs Coapplicant Income")
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')
st.pyplot(fig1)

# -----------------------------
# Chart 2: Total Income vs Loan Amount
# -----------------------------
st.subheader("📈 Total Income vs Loan Amount")
total_income = applicant_income + coapplicant_income
loan_amount_full = loan_amount * 1000

fig2, ax2 = plt.subplots()
bars2 = ax2.bar(['Total Income', 'Loan Amount'], [total_income, loan_amount_full], color=['gold', 'salmon'])
ax2.set_ylabel("Amount (₹)")
ax2.set_title("Total Income vs Loan Amount")
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')
st.pyplot(fig2)

# -----------------------------
# Chart 3: Loan Approval vs Rejection (Prediction Visualization)
# -----------------------------
st.subheader("🔍 Loan Status Prediction")
labels = ['Approved', 'Rejected']
sizes = [1, 0] if prediction == 1 else [0, 1]
colors = ['green', 'red']

fig3, ax3 = plt.subplots()
ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax3.axis('equal')
st.pyplot(fig3)

# -----------------------------
# PDF Download Section
# -----------------------------
st.subheader("📄 Download Loan Prediction Report")

def generate_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)

    text.textLine("🏦 Loan Prediction Report")
    text.textLine("--------------------------------------")
    text.textLine(f"Gender: {'Male' if gender == 1 else 'Female'}")
    text.textLine(f"Married: {'Yes' if married == 1 else 'No'}")
    text.textLine(f"Dependents: {dependents}")
    text.textLine(f"Education: {'Graduate' if education == 1 else 'Not Graduate'}")
    text.textLine(f"Self Employed: {'Yes' if self_employed == 1 else 'No'}")
    text.textLine(f"Applicant Income: ₹{applicant_income}")
    text.textLine(f"Coapplicant Income: ₹{coapplicant_income}")
    text.textLine(f"Total Income: ₹{total_income}")
    text.textLine(f"Loan Amount: ₹{loan_amount_full}")
    text.textLine(f"Loan Term: {loan_term} days")
    text.textLine(f"Credit History: {'Good' if credit_history == 1 else 'Poor'}")
    area_text = {2: "Urban", 1: "Semiurban", 0: "Rural"}[property_area]
    text.textLine(f"Property Area: {area_text}")
    text.textLine("--------------------------------------")
    text.textLine(f"Loan Prediction Result: {result}")
    
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

pdf_file = generate_pdf()

st.download_button(
    label="📥 Download PDF Report",
    data=pdf_file,
    file_name="loan_prediction_report.pdf",
    mime="application/pdf"
)


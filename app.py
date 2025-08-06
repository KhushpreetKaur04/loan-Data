import streamlit as st
import joblib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

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


# ----------------- Visual Section -----------------

# 🎨 Set clean banking-themed background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.freepik.com/free-vector/futuristic-digital-rupee-money-concept-background_34426968.htm#fromView=keyword&page=1&position=28&uuid=1fe0b7eb-371b-4847-a8f1-7f76aae1f01c&query=Banking+Background");
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    .highlight-box {
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Convert numerical inputs for display
gender_text = "Male" if gender == 1 else "Female"
region_text = ["Rural", "Semiurban", "Urban"][property_area]

# 📦 Display summary boxes: Gender, Dependents, Region
st.markdown("### 📌 Applicant Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="highlight-box">Gender<br>{gender_text}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="highlight-box">Dependents<br>{dependents}</div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="highlight-box">Region<br>{region_text}</div>', unsafe_allow_html=True)

# 📊 Income vs Loan Amount Chart
st.markdown("### 💰 Income vs Loan Amount")

loan_amount_actual = loan_amount   # convert to actual ₹
categories = ['Applicant Income', 'Coapplicant Income', 'Loan Amount']
values = [applicant_income, coapplicant_income, loan_amount_actual]

fig, ax = plt.subplots()
bars = ax.bar(categories, values, color=['#4a90e2', '#50e3c2', '#f5a623'])
ax.set_ylabel("Amount (in ₹)")
ax.set_title("Applicant & Co-applicant Income vs Loan Amount")

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'₹{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom')

st.pyplot(fig)

# ✅ Loan Status Visual
if 'prediction' in locals():
    st.markdown("### 📋 Loan Approval Visual")
    if prediction == 1:
        st.success("Congratulations! Your loan is likely to be approved.")
        st.image("https://cdn-icons-png.flaticon.com/512/845/845646.png", width=120)  # tick icon
    else:
        st.error("Sorry, your loan may not be approved.")
        st.image("https://cdn-icons-png.flaticon.com/512/463/463612.png", width=120)  # cross icon

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import base64
import os

# 📥 Generate PDF report after prediction
if 'prediction' in locals():
    st.markdown("---")
    st.markdown("### 📄 Download Prediction Report")

    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph("🏦 Loan Approval Prediction Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Details
        elements.append(Paragraph(f"<b>Prediction:</b> {'Loan Approved ✅' if prediction == 1 else 'Loan Rejected ❌'}", styles['Normal']))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>Gender:</b> {gender_text}", styles['Normal']))
        elements.append(Paragraph(f"<b>Dependents:</b> {dependents}", styles['Normal']))
        elements.append(Paragraph(f"<b>Region:</b> {region_text}", styles['Normal']))
        elements.append(Paragraph(f"<b>Applicant Income:</b> ₹{applicant_income}", styles['Normal']))
        elements.append(Paragraph(f"<b>Coapplicant Income:</b> ₹{coapplicant_income}", styles['Normal']))
        elements.append(Paragraph(f"<b>Loan Amount:</b> ₹{loan_amount_actual}", styles['Normal']))
        elements.append(Spacer(1, 12))

        # Save PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf()
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )





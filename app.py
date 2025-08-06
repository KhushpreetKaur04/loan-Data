
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Background image (replace URL or use local path logic if needed)
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1563013544-824ae1b704d3");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Approval Prediction App")

# Sidebar inputs
st.sidebar.header("Applicant Details")

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

# Encode categorical values
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area_code = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Predict
if st.sidebar.button("Predict Loan Status"):
    input_data = np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_term, credit_history, property_area_code]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    # Show result
    st.subheader("Prediction Result:")
    st.success(result if prediction == 1 else "")
    st.error(result if prediction == 0 else "")

    # Chart 1: Bar Chart
    st.subheader("📊 Applicant Income vs Loan Amount")
    df_bar = pd.DataFrame({
        'Category': ['Applicant Income', 'Coapplicant Income', 'Loan Amount (x1000)'],
        'Value': [applicant_income, coapplicant_income, loan_amount]
    })
    st.bar_chart(df_bar.set_index("Category"))

    # Chart 2: Histogram
    st.subheader("📉 Income Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist([applicant_income, coapplicant_income], bins=5, label=["Applicant", "Coapplicant"], alpha=0.7)
    ax1.set_xlabel("Income")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    st.pyplot(fig1)

    # Chart 3: Pie chart
    st.subheader("🎯 Loan Approval Chance")
    fig2, ax2 = plt.subplots()
    ax2.pie([prediction, 1 - prediction],
            labels=['Approved', 'Rejected'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['green', 'red'])
    ax2.axis('equal')
    st.pyplot(fig2)

    # Generate PDF
    def create_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("🏦 Loan Approval Prediction Report", styles['Title']))
        elements.append(Paragraph(f"Loan Status: {result}", styles['Heading2']))
        elements.append(Paragraph(f"Applicant Income: {applicant_income}", styles['Normal']))
        elements.append(Paragraph(f"Coapplicant Income: {coapplicant_income}", styles['Normal']))
        elements.append(Paragraph(f"Loan Amount: {loan_amount}", styles['Normal']))
        elements.append(Paragraph(f"Loan Term: {loan_term}", styles['Normal']))
        elements.append(Paragraph(f"Credit History: {credit_history}", styles['Normal']))
        elements.append(Paragraph(f"Property Area: {property_area}", styles['Normal']))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    # PDF download
    st.download_button(
        label="📄 Download Loan Report PDF",
        data=create_pdf(),
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )


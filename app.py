import streamlit as st
import joblib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ✅ Set background image (Update the URL below with your own image link)
background_image_url = "https://th.bing.com/th/id/R.496329cae6ac855e1bf6ebf616b0ebeb?rik=3xLzxHZSu10aSw&riu=http%3a%2f%2fincomeseries.com%2fwp-content%2fuploads%2f2023%2f08%2fbanking-conclusion.webp&ehk=9S7BkLjv%2fGW8eTcUyAIXDiujl34Pu0UK%2f2yA6qPJYM0%3d&risl=&pid=ImgRaw&r=0"  # <-- Replace with your image URL
page_bg_img = f'''
<style>
.stApp {{
    background-image: url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

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

# ✅ Loan Status 
if 'prediction' in locals():
    st.markdown("### 📋 Loan Approval Visual")
    if prediction == 1:
        st.success("Congratulations! Your loan is likely to be approved.")
        st.image("https://cdn-icons-png.flaticon.com/512/845/845646.png", width=120)
    else:
        st.error("Sorry, your loan may not be approved.")
        st.image("https://cdn-icons-png.flaticon.com/512/463/463612.png", width=120)

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

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















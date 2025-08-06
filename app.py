import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------- Set Background Image --------------------
def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.pexels.com/photos/4386436/pexels-photo-4386436.jpeg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg()

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

    # -------------------- Charts --------------------
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    st.markdown("### 📊 Financial Visualization")

    # Chart 1: Income Comparison Bar
    fig1, ax1 = plt.subplots(figsize=(6, 2.8))
    df_income = pd.DataFrame({
        'Income Type': ['Applicant', 'Coapplicant'],
        'Amount': [applicant_income, coapplicant_income]
    })
    sns.barplot(data=df_income, x='Amount', y='Income Type', ax=ax1, palette='Set2')
    ax1.set_title("Income Comparison", fontsize=13)
    ax1.set_xlabel("Income (₹)", fontsize=11)
    ax1.set_ylabel("")
    st.pyplot(fig1)

    # Chart 2: Income vs Loan Amount
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    total_income = applicant_income + coapplicant_income
    sns.scatterplot(x=[total_income], y=[loan_amount], s=200, color="darkblue", edgecolor='white')
    ax2.set_title("Total Income vs Loan Amount", fontsize=13)
    ax2.set_xlabel("Total Income (₹)", fontsize=11)
    ax2.set_ylabel("Loan Amount (₹)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig2)

    # -------------------- PDF Report --------------------
    from io import BytesIO
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    import base64

    def generate_pdf(prediction_text):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("🏦 Final Loan Prediction Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Prediction Result: {prediction_text}", styles["Heading2"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Applicant Income: ₹{applicant_income}", styles["Normal"]))
        elements.append(Paragraph(f"Coapplicant Income: ₹{coapplicant_income}", styles["Normal"]))
        elements.append(Paragraph(f"Loan Amount: ₹{loan_amount}", styles["Normal"]))
        elements.append(Paragraph(f"Loan Term: {loan_term} days", styles["Normal"]))
        elements.append(Paragraph(f"Credit History: {'Good' if credit_history == 1 else 'Bad'}", styles["Normal"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("📌 This result is based on machine learning prediction using your financial profile. Please consult with the bank for final approval.", styles["Normal"]))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf = generate_pdf(result)
    b64 = base64.b64encode(pdf.read()).decode()

    st.download_button(
        label="📄 Download Final Loan Prediction Report (PDF)",
        data=base64.b64decode(b64),
        file_name="Loan_Prediction_Report.pdf",
        mime="application/pdf"
    )





























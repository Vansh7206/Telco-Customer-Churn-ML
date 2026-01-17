import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model and columns
model = joblib.load("models/churn_logistic_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

THRESHOLD = 0.3

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Telecom Customer Churn Prediction")
st.write("Predict churn risk to support proactive retention decisions.")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'dataset', 'Telco_customer_churn.xlsx')
    return pd.read_excel(data_path)

df = load_data()
st.write('##  📊 Dataset Preview')
st.dataframe(df.head(11))

# User Inputs
st.subheader("ℹ️ Customer Information")

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

# Build input dataframe
input_data = {
    "Tenure Months": tenure,
    "Monthly Charges": monthly_charges,
    "Contract": contract,
    "Internet Service": internet_service,
    "Payment Method": payment_method,
    "Senior Citizen": senior,
    "Partner": partner,
    "Dependents": dependents
}

input_df = pd.DataFrame([input_data])

# One-hot encode
input_encoded = pd.get_dummies(input_df)

# Align with training columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("Predict Churn Risk"):
    churn_prob = model.predict_proba(input_encoded)[0][1]
    churn_pred = "Yes" if churn_prob >= THRESHOLD else "No"

    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{churn_prob:.2%}")

    if churn_pred == "Yes":
        st.error("⚠️ High Risk of Churn")
        st.write("Recommended Action: Engage customer with retention offers.")
    else:
        st.success("✅ Low Risk of Churn")
        st.write("Recommended Action: No immediate intervention required.")

    st.caption(f"Decision Threshold Used: {THRESHOLD}")

st.markdown("---")
st.info(' Made by Vansh Chandan ⚙️🦾🧑🏻‍💻')


import streamlit as st
import requests

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")

st.sidebar.info(
    "Features V1–V28 are anonymized principal components from the original dataset. "
    "Their exact meanings are not public, but they are derived from transaction details."
)

st.write("Enter transaction details below:")

features = {}
for i in range(1, 29):
    features[f"V{i}"] = st.number_input(
        f"V{i}",
        value=0.0,
        help="Anonymized feature from PCA. Original meaning is not public."
    )
features["Amount"] = st.number_input("Transaction Amount (USD)", value=0.0)

if st.button("Predict Fraud Probability"):
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=features
        )
        if response.status_code == 200:
            proba = response.json()["fraud_probability"]
            st.success(f"Fraud Probability: {proba:.2%}")
            if proba > 0.5:
                st.error("⚠️ High risk of fraud!")
            else:
                st.info("✅ Low risk of fraud.")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Could not connect to API: {e}") 
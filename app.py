import streamlit as st
import pandas as pd
import pickle

with open("model/logistic_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction App")

st.write("Upload customer data and predict churn using Logistic Regression.")


uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    # Read uploaded data
    data = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.write(data.head())

    # Scale features
    X_scaled = scaler.transform(data)

    # Predict churn
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Show results
    results = pd.DataFrame({
        "Churn Prediction": predictions,
        "Churn Probability": probabilities
    })

    st.write("### Prediction Results")
    st.write(results)

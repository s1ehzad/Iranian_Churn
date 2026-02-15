import streamlit as st
import pandas as pd
import pickle

st.title("📊 Customer Churn Prediction App")

st.write("Upload customer data and predict churn using Logistic Regression, Naive Bayes, Decision Tree, KNN, or Random Forest.")

# Sidebar model selector
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Logistic Regression", "Naive Bayes", "Decision Tree", "KNN", "Random Forest")
)

# Load the appropriate model and scaler
if model_choice == "Logistic Regression":
    with open("model/logistic_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

elif model_choice == "Naive Bayes":
    with open("model/naive_bayes_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler_nb.pkl", "rb") as f:
        scaler = pickle.load(f)

elif model_choice == "Decision Tree":
    with open("model/decision_tree_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler_dt.pkl", "rb") as f:
        scaler = pickle.load(f)

elif model_choice == "KNN":
    with open("model/knn_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler_knn.pkl", "rb") as f:
        scaler = pickle.load(f)

elif model_choice == "Random Forest":
    with open("model/random_forest_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler_rf.pkl", "rb") as f:
        scaler = pickle.load(f)

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    # Read uploaded data
    data = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.write(data.head())

    # Drop target column if present
    if "Churn" in data.columns:
        data = data.drop("Churn", axis=1)

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

    # Download option
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name=f"churn_predictions_{model_choice.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

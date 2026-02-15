import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.title("📊 Customer Churn Prediction App")

st.write("Upload customer data and predict churn using Logistic Regression, Naive Bayes, Decision Tree, KNN, Random Forest, or XGBoost.")

# Sidebar model selector
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Logistic Regression", "Naive Bayes", "Decision Tree", "KNN", "Random Forest", "XGBoost")
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

elif model_choice == "XGBoost":
    with open("model/xgboost_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler_xgb.pkl", "rb") as f:
        scaler = pickle.load(f)

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    # Read uploaded data
    data = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.write(data.head())

    # Separate features and target if available
    if "Churn" in data.columns:
        X = data.drop("Churn", axis=1)
        y_true = data["Churn"]
    else:
        X = data
        y_true = None

    # Scale features
    X_scaled = scaler.transform(X)

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

    # If true labels are available, show evaluation metrics
    if y_true is not None:
        st.write("### Evaluation Metrics")

        accuracy = accuracy_score(y_true, predictions)
        auc = roc_auc_score(y_true, probabilities)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        mcc = matthews_corrcoef(y_true, predictions)

        metrics_df = pd.DataFrame({
            "Accuracy": [accuracy],
            "AUC": [auc],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
            "MCC": [mcc]
        })

        st.table(metrics_df)

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true, predictions)
        st.write(cm)

        st.write("### Classification Report")
        report = classification_report(y_true, predictions, output_dict=False)
        st.text(report)

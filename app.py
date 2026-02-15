import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import model functions
from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost_model import train_xgboost

st.set_page_config(page_title="Heart Disease Classification App", layout="wide")

st.title("💓 Heart Disease Classification using Machine Learning")
st.write("This application compares multiple ML models for heart disease prediction.")

st.subheader("📥 Download Sample Test Dataset")

# Generate sample dataset template
def generate_sample_csv():
    sample_data = {
        "age": [52, 45],
        "sex": [1, 0],
        "cp": [0, 2],
        "trestbps": [125, 130],
        "chol": [212, 250],
        "fbs": [0, 1],
        "restecg": [1, 0],
        "thalach": [168, 150],
        "exang": [0, 1],
        "oldpeak": [1.0, 2.3],
        "slope": [2, 1],
        "ca": [2, 0],
        "thal": [3, 2],
        "target": [1, 0]
    }

    return pd.DataFrame(sample_data)

sample_df = generate_sample_csv()

csv = sample_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sample Test CSV",
    data=csv,
    file_name="sample_test.csv",
    mime="text/csv",
)


# -----------------------------
# CSV Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    data_path = uploaded_file
    st.success("Dataset uploaded successfully!")

    # Model selection
    model_option = st.selectbox(
        "Select Classification Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost",
        ),
    )

    # Train selected model
    if model_option == "Logistic Regression":
        model, metrics = train_logistic_regression(data_path)
    elif model_option == "Decision Tree":
        model, metrics = train_decision_tree(data_path)
    elif model_option == "KNN":
        model, metrics = train_knn(data_path)
    elif model_option == "Naive Bayes":
        model, metrics = train_naive_bayes(data_path)
    elif model_option == "Random Forest":
        model, metrics = train_random_forest(data_path)
    elif model_option == "XGBoost":
        model, metrics = train_xgboost(data_path)

    # -----------------------------
    # Display Metrics
    # -----------------------------
    st.subheader("📊 Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(metrics["Accuracy"], 4))
    col2.metric("AUC", round(metrics["AUC"], 4))
    col3.metric("Precision", round(metrics["Precision"], 4))

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", round(metrics["Recall"], 4))
    col5.metric("F1 Score", round(metrics["F1 Score"], 4))
    col6.metric("MCC", round(metrics["MCC"], 4))

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("📌 Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.info("Please upload the heart disease CSV file to proceed.")

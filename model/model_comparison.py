import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd

from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost_model import train_xgboost


def generate_comparison_table(data_path):
    models = {
        "Logistic Regression": train_logistic_regression,
        "Decision Tree": train_decision_tree,
        "KNN": train_knn,
        "Naive Bayes": train_naive_bayes,
        "Random Forest": train_random_forest,
        "XGBoost": train_xgboost
    }

    results = []

    for model_name, train_func in models.items():
        _, metrics = train_func(data_path)

        results.append({
            "ML Model": model_name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "AUC": round(metrics["AUC"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1 Score": round(metrics["F1 Score"], 4),
            "MCC": round(metrics["MCC"], 4)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = generate_comparison_table("data/heart.csv")
    print("\nModel Comparison Table:\n")
    print(df)

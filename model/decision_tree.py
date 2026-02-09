from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

from model.preprocessing import load_and_preprocess_data


def train_decision_tree(data_path):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    # Initialize Decision Tree model
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    return model, metrics


if __name__ == "__main__":
    model, metrics = train_decision_tree("data/heart.csv")

    print("Decision Tree Performance:")
    for key, value in metrics.items():
        print(f"{key}: \n{value}\n")

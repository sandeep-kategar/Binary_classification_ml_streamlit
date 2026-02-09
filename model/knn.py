from sklearn.neighbors import KNeighborsClassifier
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


def train_knn(data_path):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    # Initialize KNN model
    model = KNeighborsClassifier(
        n_neighbors=5,
        metric="minkowski"
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
    model, metrics = train_knn("data/heart.csv")

    print("KNN Performance:")
    for key, value in metrics.items():
        print(f"{key}: \n{value}\n")

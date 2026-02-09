# Binary Classification for Machine Learning Assignement 2
This project implements multiple machine learning classification models on the binary classification and deploys the results using a Streamlit web application.

## Technologies Used
- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy

Dataset Preprocessing
The Heart Disease dataset was preprocessed to ensure data quality and improve the performance of machine learning models. Initially, the dataset was loaded using the Pandas library. Missing values present in the dataset were handled by removing incomplete records to maintain consistency across all features.
The dataset was then divided into input features and the target variable, where the target represents the presence or absence of heart disease. An 80–20 train–test split was performed to evaluate model performance on unseen data. Stratified sampling was used during the split to preserve the original class distribution in both training and testing sets.
Since some of the implemented machine learning algorithms, such as Logistic Regression and K-Nearest Neighbors, are sensitive to feature scales, feature standardization was applied using the StandardScaler technique. This transformation ensured that all input features have zero mean and unit variance, thereby improving model convergence and performance.
The same preprocessing pipeline was consistently applied across all classification models to ensure a fair and reliable comparison of evaluation metrics.

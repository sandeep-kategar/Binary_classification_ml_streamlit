🫀 Heart Disease Classification using Machine Learning

BITS Pilani – M.Tech (AIML) – Machine Learning Assignment 2

**Live Streamlit Application:**
https://binaryclassificationmlapp-dza5vhgtlebsc5ltigymzv.streamlit.app/

**1) Problem Statement**
Heart disease is one of the leading causes of mortality worldwide. Early detection of heart disease can significantly improve patient outcomes and reduce medical risks.
The objective of this project is to build and compare multiple machine learning classification models to predict the presence or absence of heart disease based on clinical attributes.
This project also demonstrates end-to-end ML deployment by integrating trained models into an interactive Streamlit web application.

**2) Dataset Description**
The Heart Disease dataset was obtained from a public repository (Kaggle). The dataset contains clinical and demographic information of patients used to predict the presence of heart disease.
Dataset Characteristics:
Type: Binary Classification
Target Variable:
0 → No heart disease
1 → Presence of heart disease
Number of features: 13
Number of instances: ~900+
Important Features:
Age
Sex
Chest pain type
Resting blood pressure
Serum cholesterol
Maximum heart rate
ST depression
Number of major vessels
Thalassemia
The dataset includes a mix of numerical and categorical features, making it suitable for evaluating various classification algorithms.

**3️) Dataset Preprocessing**
The dataset was preprocessed to ensure data quality and improve model performance.
Missing values were removed to maintain consistency.
Features and target variable were separated.
An 80–20 train-test split was performed.
Stratified sampling was used to preserve class distribution.
Feature scaling (StandardScaler) was applied to normalize input features.
The same preprocessing pipeline was used across all models to ensure fair comparison.

**4️) Machine Learning Models Implemented**
The following six classification models were implemented:
Logistic Regression
Decision Tree
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Random Forest (Ensemble – Bagging)
XGBoost (Ensemble – Boosting)
Each model was evaluated using the following metrics:
Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

**5️) Model Comparison Table**
ML Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8098	0.9298	0.7619	0.9143	0.8312	0.6309
Decision Tree	0.8732	0.9326	0.8624	0.8952	0.8785	0.7465
KNN	0.8634	0.9629	0.8738	0.8571	0.8654	0.7269
Naive Bayes	0.8293	0.9043	0.8070	0.8762	0.8402	0.6602
Random Forest	0.9220	0.9708	0.9009	0.9524	0.9259	0.8450
XGBoost	0.8976	0.9692	0.8818	0.9238	0.9023	0.7957

**6️) Model-wise Observations**
**Logistic Regression**
Served as a strong baseline model. It achieved high recall, indicating good detection of heart disease cases. However, as a linear model, it may not fully capture complex relationships among features.

**Decision Tree**
Improved accuracy by modeling non-linear relationships. Performance was balanced across precision and recall.

**KNN**
Demonstrated strong class separability with high AUC. Performance is sensitive to feature scaling and choice of k-value.

**Naive Bayes**
Performed moderately well despite assuming feature independence. Its probabilistic approach makes it computationally efficient.

**Random Forest**
Achieved the best overall performance among all models. By combining multiple decision trees, it reduced variance and improved generalization. Controlled hyperparameters helped reduce overfitting while maintaining high predictive performance.

**XGBoost**
Performed strongly with high AUC and balanced precision-recall values. After applying regularization and limiting model complexity, overfitting was reduced compared to the initial configuration, resulting in improved generalization.

**7️) Overall Conclusion**
Ensemble learning techniques (Random Forest and XGBoost) outperformed traditional classifiers, demonstrating their effectiveness in handling complex medical datasets.
Logistic Regression provided a reliable baseline, while tree-based and ensemble methods significantly enhanced predictive capability.
The project highlights the importance of:
Proper preprocessing
Fair evaluation metrics
Hyperparameter tuning
Overfitting control
End-to-end ML deployment

**8️) Streamlit Web Application**
An interactive Streamlit web application was developed with the following features:
CSV dataset upload option
test sample download option
Model selection dropdown
Display of evaluation metrics
Confusion matrix visualization
The application enables users to dynamically compare model performance on uploaded datasets.

**9️) Technologies Used**
Python
Scikit-learn
XGBoost
Pandas
NumPy
Matplotlib
Seaborn
Streamlit

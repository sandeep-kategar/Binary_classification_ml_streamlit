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

**Model Comparison Table**
| ML Model            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.8098   | 0.9298 | 0.7619    | 0.9143 | 0.8312   | 0.6309 |
| Decision Tree       | 0.8732   | 0.9326 | 0.8624    | 0.8952 | 0.8785   | 0.7465 |
| KNN                 | 0.8634   | 0.9629 | 0.8738    | 0.8571 | 0.8654   | 0.7269 |
| Naive Bayes         | 0.8293   | 0.9043 | 0.8070    | 0.8762 | 0.8402   | 0.6602 |
| Random Forest       | 0.9902   | 0.9996 | 0.9813    | 1.0000 | 0.9906   | 0.9807 |
| XGBoost             | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000   | 1.0000 |

Model-wise Observations
1) Logistic Regression
Logistic Regression achieved moderate accuracy (80.98%) with a high recall (91.43%), indicating strong ability to identify positive heart disease cases. However, comparatively lower precision suggests some false positives. As a linear model, it may not fully capture complex feature interactions.

2) Decision Tree
Decision Tree improved performance significantly with 87.32% accuracy and balanced precision–recall values. The model effectively captured non-linear relationships but may be prone to overfitting without depth control.

3) K-Nearest Neighbors (KNN)
KNN achieved strong AUC (96.29%), indicating excellent class separation capability. However, performance is sensitive to feature scaling and choice of k-value.

4) Naive Bayes
Naive Bayes showed moderate performance with 82.93% accuracy. Although it assumes feature independence (which may not hold fully in medical data), it still performed competitively due to probabilistic modeling.

5) Random Forest (Ensemble Model)
Random Forest achieved very high accuracy (99.02%) and near-perfect AUC (0.9996). The ensemble approach reduced variance and captured complex interactions between features, significantly improving predictive performance.

6) XGBoost (Ensemble Boosting Model)
XGBoost achieved perfect performance metrics on the test dataset. This indicates extremely strong predictive capability. However, such high performance may suggest potential overfitting, especially if the dataset is relatively small or highly separable.

Overall Conclusion:
Among all models, ensemble methods (Random Forest and XGBoost) demonstrated superior performance compared to traditional classifiers. This highlights the effectiveness of ensemble learning in handling complex medical classification problems. Logistic Regression served as a strong baseline model, while tree-based methods significantly improved predictive capability.

# Iranian_Churn
Iranian Churn
Problem Statement
The goal of this project is to build predictive models that can identify customers likely to churn, enabling proactive retention strategies. We implemented and compared six machine learning classifiers to evaluate their performance on the churn dataset.

Dataset Description
The dataset used is customer churn.csv, which contains customer attributes (demographics, usage patterns, service details) along with the target variable Churn (binary: 1 = churn, 0 = no churn).
- Features: Multiple numerical and categorical variables representing customer behavior and service usage.
Variable Name	Role	Type	Demographic	Description	Units	Missing Values
Call Failure	Feature	Integer				no
Complains	Feature	Binary				no
Subscription Length	Feature	Integer				no
Charge Amount	Feature	Integer				no
Seconds of Use	Feature	Integer				no
Frequency of use	Feature	Integer				no
Frequency of SMS	Feature	Integer				no
Distinct Called Numbers	Feature	Integer				no
Age Group	Feature	Integer	Age			no
Tariff Plan	Feature	Integer				no
Status	Feature	Binary				no
Age	Feature	Integer	Age			no
Customer Value	Feature	Continuous				no

- Target: Churn column indicating whether the customer left the service.

Models Used and Evaluation Metrics
We implemented the following models:
- Logistic Regression
- Decision Tree
- K‑Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

ML Model Name	Accuracy	AUC Score	Precision	Recall	F1 Score	MCC
Logistic Regression	0.896825	0.920790	0.840000	0.424242	0.563758	0.550905
Decision Tree	0.926984	0.867336	0.773196	0.757576	0.765306	0.722131
kNN	0.952381	0.967671	0.863158	0.828283	0.845361	0.817475
Naive Bayes	0.738095	0.898609	0.364754	0.898990	0.518950	0.453553
Random Forest	0.963492	0.988720	0.904255	0.858586	0.880829	0.859693
XGBoost	0.971429	0.992220	0.900990	0.919192	0.910000	0.893084


ML Model Name	Observation about Model Performance
Logistic Regression	Provides a strong baseline but recall is relatively low, meaning it misses many churn cases.
Decision Tree	Balanced precision and recall, but slightly lower AUC compared to ensemble methods.
kNN	Performs well overall, but computationally heavier and sensitive to scaling.
Naive Bayes	Very high recall but poor precision, leading to many false positives.
Random Forest	Strong performance across all metrics, robust and less prone to overfitting
XGBoost	Best overall performer with highest accuracy, AUC, and balanced precision/recall.


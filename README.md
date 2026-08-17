# Telco Customer Churn — Classification Model Comparison

## a. Problem Statement

Customer churn — when a customer stops using a company's service — is a major
cost driver for subscription-based businesses like telecom providers. This
project builds and compares multiple classification models to predict whether
a telecom customer will churn (leave the service) based on their account
details, service subscriptions, and billing information. The goal is to
identify which model best distinguishes churners from non-churners, so a
business could use it to flag at-risk customers for retention efforts.

## b. Dataset Description

- Source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- Size: 7,043 customers, 20 features + 1 target column (`Churn`)
- Target: `Churn` — binary (`Yes` / `No`), encoded as 1/0
- Class balance: ~73.5% No Churn, ~26.5% Churn (moderately imbalanced)
- Features: A mix of numeric (tenure, MonthlyCharges, TotalCharges,
  SeniorCitizen) and categorical (gender, Contract, InternetService,
  PaymentMethod, and various service subscriptions like OnlineSecurity,
  TechSupport, StreamingTV, etc.)
- Preprocessing:
  - Dropped `customerID` (identifier, not predictive)
  - `TotalCharges` had 11 blank values, all corresponding to customers with
    0 tenure (new signups) — filled with 0, since that's the logically
    correct value rather than a random imputation
  - One-hot encoded all categorical columns (`drop_first=True`)
  - Scaled numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`,
    `SeniorCitizen`) using `StandardScaler`, fit only on the training set
  - 80/20 train-test split, stratified on `Churn` to preserve class balance

## c. GitHub Repository Link

https://github.com/SahilBhragudev/ML-Classification-Streamlit

## d. Models Used

- Logistic Regression
- Decision Tree Classifier
- k Nearest Neighbour  Classifier
- Naive Bayes Classifier
- Ensemble Model (Random Forest)

### Comparison Table

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression      | 0.805    | 0.842 | 0.658     | 0.551  | 0.600 | 0.475 |
| Decision Tree            | 0.733    | 0.656 | 0.497     | 0.492  | 0.495 | 0.313 |
| kNN                      | 0.762    | 0.787 | 0.553     | 0.545  | 0.549 | 0.388 |
| Naive Bayes              | 0.659    | 0.809 | 0.430     | 0.866  | 0.574 | 0.399 |
| Random Forest (Ensemble) | 0.785    | 0.816 | 0.621     | 0.487  | 0.546 | 0.413 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This model is the best overall performer across Accuracy, AUC, and MCC metrics. The linear decision boundary works well here, likely because features like Contract type and tenure have a fairly linear relationship with churn probability. |
| Decision Tree | This model is the Weakest performer across nearly every metric. This may be due to overfitting on the training data since no depth limit or pruning was applied, which hurts the generalization on the test data set. |
| kNN | This model delivered middling performance across the various metrics. This is due to the fact that distance-based classification is sensitive to the large number of one-hot encoded columns, which can lessen the impact of more informative numeric features. |
| Naive Bayes | This model had lowest accuracy and precision, but had the highest recall (0.87) and second-highest AUC. It aggressively flags customers as likely churners, catching most true churners at the cost of many false positives. This might be helpful in some businesses that focus more on not missing at-risk customers over minimizing false alarms.|
| Random Forest (Ensemble) | This was the second-best model overall, close to Logistic Regression on AUC and MCC metrics. It Slightly underperforms Logistic Regression here, which is a bit odd for ensembles — may be  because the relationships in this dataset are fairly linear and don't benefit heavily from the non-linear splits Random Forest captures. |
| **Overall Winner for your dataset?** | **Logistic Regression** — Due to highest Accuracy, AUC, and MCC among all five models, making it the most reliable overall performer on this dataset. "Though Naive Bayes may be preferable if catching every possible churner matters more than overall precision."|

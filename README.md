# Credit Risk Assessment with Machine Learning

This project implements a **credit risk assessment system** using Python and popular machine learning libraries.  
The system predicts the **probability of loan default (PD)** and computes **expected loss (EL)** for each borrower using financial data from the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).

---

## Dataset
- Source: Kaggle → [`laotse/credit-risk-dataset`](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  
- Loaded using [`kagglehub`](https://github.com/Kaggle/kagglehub).  
- Target column: **`loan_status`**  
  - `1` → Default  
  - `0` → Non-default  

---

## Features of the Project
- **Data Preprocessing**
  - Missing value imputation (median for numeric, mode for categorical).
  - Label encoding (`OrdinalEncoder`) for categorical variables.
  - Standardization (`StandardScaler`) for numeric variables.
  - Stratified train-test splitting using **StratifiedKFold**.

- **Models Implemented**
  - K-Nearest Neighbors (KNN)  
  - Logistic Regression (LogReg)  
  - XGBoost (if installed)  

- **Evaluation**
  - Cross-validation with **ROC-AUC** and **PR-AUC**.
  - ROC Curves plotted for all models.
  - Feature importance (XGBoost gain or permutation importance).

- **Credit Risk Principles**
  - **Probability of Default (PD):** Predicted by the model.  
  - **Expected Loss (EL):** Computed as  
    \[
    EL = PD \times LGD \times EAD
    \]  
    where LGD (Loss Given Default) and EAD (Exposure at Default) are constants or provided fields.

---

## Visualizations
The notebook / script produces several plots:

- **ROC Curves** (for KNN, Logistic Regression, XGBoost)  
- **Feature Importance** (bar chart for top predictors)  
- **Numeric Feature Distributions** by loan default status (KDE plots)  
- **Categorical Feature Impact** (countplots)  
- **Scatter Plots** (relationship between numeric pairs colored by loan status)  
- **Correlation Heatmap** with values inside squares (shows strongest predictors of default)  

---

## Requirements
Install the dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn kagglehub

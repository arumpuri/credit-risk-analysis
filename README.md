# Credit Risk Analysis Pipeline

This repository contains a comprehensive machine learning pipeline for credit risk analysis. The goal is to predict the probability of default for borrowers and calculate the expected loss on loans, assuming a recovery rate of 10%. The pipeline includes advanced feature engineering, preprocessing, model training, evaluation, and interpretability tools.

## Features

- **Feature Engineering**: 
  - Derived features such as `debt_to_income`, `loan_to_debt_ratio`, and `credit_utilization`.
  - Binning of continuous variables like `fico_score` for better interpretability.
  
- **Preprocessing**:
  - Robust scaling for numeric features.
  - One-hot encoding for categorical features.
  - Outlier handling using IQR-based capping.

- **Models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - Stacking Ensemble (combining Logistic Regression, Random Forest, and XGBoost)

- **Evaluation Metrics**:
  - ROC AUC
  - Precision-Recall AUC
  - Confusion Matrix
  - Feature Importance Analysis
  - SHAP-based interpretability

- **Expected Loss Calculation**:
  - Function to estimate the expected loss on a loan based on the predicted probability of default and recovery rate.

## Requirements

- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `matplotlib`, `seaborn`

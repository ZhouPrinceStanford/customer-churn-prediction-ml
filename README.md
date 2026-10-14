# Customer Churn Prediction Using Machine Learning

## Project Overview

This project uses machine learning to predict whether a telecommunications customer is likely to leave a service. The objective is to help a business identify customers who may benefit from targeted retention strategies.

## Dataset

- **Observations:** 7,043 customers
- **Features:** 21 columns in the original dataset
- **Target:** `Churn` — whether a customer left the service
- **Data preparation:** Blank or whitespace-only values are handled, and `TotalCharges` is converted to a numeric variable.

## Methodology

1. Inspect and clean the dataset.
2. Separate predictors from the target variable.
3. Split the data into training and test sets using stratified sampling and a fixed random seed.
4. Preprocess numerical and categorical variables.
5. Tune and compare machine-learning models using cross-validation.
6. Select a classification threshold using out-of-fold training predictions.
7. Evaluate the selected model on a held-out test set.

The test set was kept separate from model tuning and threshold selection.

## Model and Evaluation

The selected model is an XGBoost classifier within a preprocessing pipeline.

The classification threshold is **0.30**.

| Test metric | Result |
|---|---:|
| Accuracy | 75.80% |
| Churn precision | 53.01% |
| Churn recall | 77.81% |
| Churn F1-score | 63.06% |
| ROC-AUC | 0.8477 |
| PR-AUC | 0.6629 |

### Confusion Matrix

| | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| Actual: No Churn | 777 | 258 |
| Actual: Churn | 83 | 291 |

### Interpretation

At the selected threshold, the model identifies approximately 78% of customers who churned in the test set. Churn precision is approximately 53%, meaning that some customers flagged by the model did not churn.

The threshold reflects a trade-off between identifying more potential churners and the cost of contacting customers who ultimately stay.

## Technologies

- Python
- pandas
- scikit-learn
- XGBoost
- Jupyter Notebook
- joblib

## Project Files

- `Customer Churn Prediction Using Machine Learning.ipynb` — analysis, model development, and evaluation.
- `Customer-Churn.csv` — dataset used in the analysis.
- `customer_churn_xgboost_final.pkl` — saved preprocessing and prediction pipeline, including the classification threshold.
- `customer_churn_xgboost_native.json` — native XGBoost model export.

**Note:** The native JSON model contains the XGBoost component only. Use the saved pipeline when making predictions from raw customer data because it also contains the preprocessing steps.

## Limitations

- Results are based on one held-out test split.
- Model performance may change on new customer populations or future data.
- Churn predictions should support, not replace, business judgement.
- The cost of retention campaigns and the business value of correctly identifying churners were not quantified here.

## Reproducibility

Install the required Python libraries, open the notebook, and run its cells in order. The saved pipeline was created with specific library versions; use compatible versions when loading it.

## Author

Machine Learning Portfolio Project

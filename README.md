# 📊 Customer Churn Prediction Using Machine Learning

## 🔍 Introduction

Customer churn — when existing customers stop doing business with a company — is a critical challenge in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project predicts customer churn using a real-world telecom dataset and machine learning techniques, with a focus on handling class imbalance correctly.

The project uses the **Telco Customer Churn dataset**, which contains customer demographics, account info, and service usage metrics (~26% churn rate).

---

## 🎯 Objectives

- Perform comprehensive **Data Cleaning** and **Exploratory Data Analysis (EDA)**.
- Handle missing values and convert data types where necessary (e.g., `TotalCharges`).
- Use **Label Encoding** to convert categorical variables into numerical format.
- Benchmark **SMOTE (Synthetic Minority Oversampling Technique)** against **cost-sensitive learning** for handling class imbalance.
- Train and evaluate three tree-based models: **Decision Tree**, **Random Forest**, **XGBoost**.
- Use **Stratified K-Fold Cross-Validation** for robust evaluation.
- Tune the classification decision threshold using cross-validation to avoid data leakage.
- Save the final model, encoders, and threshold using **Pickle**.
- Make predictions using the best-performing configuration.

---

## 🧪 Technologies Used

- Python
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn
- XGBoost
- Pickle
- Jupyter Notebook

---

## 📈 Workflow Summary

1. **Data Cleaning & EDA**
   - Handled missing values
   - Converted `TotalCharges` column to numeric
   - Visualized patterns in churn distribution and service usage

2. **Preprocessing**
   - Applied **Label Encoding** to categorical variables
   - No normalization applied to `MonthlyCharges`/`TotalCharges` — tree-based models split on thresholds and are scale-invariant, so this was a deliberate choice rather than an oversight

3. **Baseline Model — SMOTE + Random Forest**
   - Benchmarked Decision Tree, Random Forest, and XGBoost with **SMOTE**-balanced training data
   - Validated with **Stratified 5-Fold Cross-Validation**
   - Random Forest selected based on highest CV accuracy
   - Evaluated on a held-out test set using the default 0.5 classification threshold

4. **Improved Model — Cost-Sensitive Learning + Threshold Tuning**
   - At ~26% minority class, SMOTE is a relatively mild fix for what is a moderate (not severe) imbalance problem. Switched to **cost-sensitive learning** (`class_weight="balanced"`) to penalize misclassification of churners directly, without altering the training data.
   - Tuned the decision threshold (instead of using the default 0.5 cutoff) by sweeping values inside each fold of a **Stratified 5-Fold Cross-Validation**, then averaging the optimal threshold across folds. This avoids the data leakage that occurs when threshold selection touches the test set.
   - Final threshold applied exactly once on the untouched test set.

5. **Evaluation**
   - Compared both approaches on Precision, Recall, F1-score, and ROC-AUC for the churn class
   - The improved approach trades some precision for a significant recall gain — catching more actual churners, which is the more business-relevant outcome

6. **Model Saving**
   - Saved the trained model, label encoders, and **tuned threshold** together as `.pkl` files (the threshold isn't learned by the model, so it must be saved and applied explicitly at inference time)
   - Built a predictive system that uses `predict_proba()` plus the saved threshold rather than the model's default `.predict()`

---

## ✅ Results

### Baseline model comparison (Stratified 5-Fold CV, SMOTE-balanced data)

| Model          | Cross-Validation Accuracy (Mean) |
|----------------|-----------------------------------|
| Decision Tree  | ~78%                               |
| Random Forest  | ~85% ✅ (Selected)                  |
| XGBoost        | ~84%                               |

### Final comparison — Test set (churn class)

| Metric           | SMOTE + Random Forest | Cost-Sensitive RF + Threshold Tuning |
|-------------------|:---------------------:|:--------------------------------------:|
| Precision          | 0.581                 | 0.556                                   |
| Recall             | 0.587                 | **0.692**                               |
| F1-score           | 0.584                 | **0.617**                               |
| ROC-AUC            | 0.823                 | **0.836**                               |

Tuned decision threshold: **0.34** (averaged across 5 cross-validation folds)

ROC-AUC is identical between the cost-sensitive model with and without threshold tuning, since ROC-AUC measures ranking ability independent of the cutoff — threshold tuning only shifts the precision/recall trade-off, while the underlying improvement in class separation comes from cost-sensitive learning itself.

---

## 📌 Conclusion

This project demonstrates an iterative approach to handling class imbalance: starting from a standard SMOTE + Random Forest baseline, identifying that SMOTE is not the most theoretically appropriate technique for a moderate (~26%) imbalance ratio, and improving on it with cost-sensitive learning and properly cross-validated threshold tuning. The final model improves churn recall from 58.7% to 69.2% and ROC-AUC from 0.823 to 0.836, while keeping the modeling pipeline fully leakage-free.

---

## 📁 Files Included

- `Customer_Churn_Prediction_using_ML.ipynb`: Main code notebook (baseline + improved model)
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset used
- `customer_churn_model_improved.pkl`: Trained model, feature names, and tuned threshold
- `encoders.pkl`: Label encoders used during preprocessing
- `requirements.txt`: Required libraries

---

## 🚀 How to Run This Project

```bash
# Clone the repo
git clone https://github.com/your-username/Customer_Churn_Prediction.git

# Navigate to project folder
cd Customer_Churn_Prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
```

---

## 🙋‍♂️ Author

[Vivek Negi](https://www.linkedin.com/in/vivek-singh-negi22)

---

## 📝 License

MIT License


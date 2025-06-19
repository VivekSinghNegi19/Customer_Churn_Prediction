# 📊 Customer Churn Prediction Using Machine Learning

## 🔍 Introduction

Customer churn — when existing customers stop doing business with a company — is a critical challenge in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to predict customer churn using a real-world telecom dataset and machine learning techniques.

The project uses the **Telco Customer Churn dataset**, which contains customer demographics, account info, and service usage metrics.

---

## 🎯 Objectives

- Perform comprehensive **Data Cleaning** and **Exploratory Data Analysis (EDA)**.
- Handle missing values and convert data types where necessary (e.g., `TotalCharges`).
- Use **Label Encoding** to convert categorical variables into numerical format.
- Balance the dataset using **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance.
- Train and evaluate three tree-based models:
  - **Decision Tree**
  - **Random Forest**
  - **XGBoost**
- Use **Stratified K-Fold Cross-Validation** for robust evaluation.
- Save the final model and encoder using **Pickle**.
- Make predictions using the best-performing model (Random Forest).

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

2. **Preprocessing & Feature Engineering**
   - Applied **Label Encoding** to categorical variables
   - Used **SMOTE** to balance the target classes

3. **Model Training**
   - Trained three models: Decision Tree, Random Forest, and XGBoost
   - Used **Stratified K-Fold Cross-Validation (cv=5)** to validate performance

4. **Evaluation**
   - Compared models using Accuracy, Precision, Recall, and F1-Score
   - **Random Forest** achieved the highest cross-validation accuracy

5. **Model Saving**
   - Saved the trained Random Forest model and label encoder as `.pkl` files
   - Used them to make predictions on new data

---

## ✅ Results

| Model          | Cross-Validation Accuracy (Mean) |
|----------------|------------------------------|
| Decision Tree  | ~78%                       |
| Random Forest  | ~85% ✅ (Selected)          |
| XGBoost        | ~84%                       |



---

## 📌 Conclusion

This project successfully predicted customer churn using a structured ML pipeline. After comparing three models, **Random Forest** was selected based on its accuracy and generalization ability. Using **Stratified Cross-Validation** and **SMOTE** improved model reliability and robustness.

The project demonstrates the end-to-end ML workflow, from preprocessing to deployment-ready modeling.

---

## 📁 Files Included

- `Customer_Churn_Prediction_using_ML.ipynb`: Main code notebook
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset used
- `model.pkl`: Trained Random Forest model
- `encoder.pkl`: Label encoder used during preprocessing
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
jupyter notebook notebooks/Customer_Churn_Prediction_using_ML.ipynb
```

---

## 🙋‍♂️ Author

[Vivek Negi](www.linkedin.com/in/vivek-singh-negi22)

---

## 📝 License

MIT License

# 📊 Telecom Customer Churn Prediction

### 📌 Project Overview
This project focuses on predicting customer churn for a telecom company using historical customer data. The goal is not just prediction, but decision support — identifying customers at risk of churning so that proactive retention actions can be taken.
The project is implemented as a single end-to-end Jupyter notebook, following an industry-aligned machine learning workflow.
<br>

### 🎯 Business Problem
Customer churn is costly for telecom companies. Retaining existing customers is significantly cheaper than acquiring new ones.
Business question:
Which customers are at risk of churning so that the business can intervene early?
Missing a churned customer is more expensive than unnecessarily contacting a customer who would stay.
<br>

### 🧠 Machine Learning Problem
- Type: Binary Classification
- Target Variable: Churn Label (Yes / No)
- Model: Logistic Regression (baseline, interpretable)
- Objective: Prioritize identifying churners rather than maximizing raw accuracy

### 📂 Dataset
1. Dataset: Telco Customer Churn
2. Rows: ~7,000 customers
3. Features: Demographics, services, contract details, tenure, and billing information

Post-outcome and leakage-prone columns (e.g., churn reason, churn score, identifiers) were excluded from modeling.

### 🛠️ Project Workflow
1️⃣ Problem Framing
Defined the business objective, cost of errors, and decision context.

2️⃣ Data Understanding
Reviewed all features to understand:
- Business meaning
- Predictive relevance
- Leakage risk

3️⃣ Exploratory Data Analysis (EDA)
- Target Analysis: Identified class imbalance in churn

- Univariate Analysis: Examined distributions of numerical and categorical features

- Bivariate Analysis: Analyzed key features vs churn
1. Tenure
2. Contract type
3. Monthly charges
4. Internet service

4️⃣ Data Cleaning
- Converted Total Charges to numeric
- Handled logically missing values
- Removed duplicate and leakage-prone columns

5️⃣ Feature Engineering
- Selected relevant modeling features
- Dropped identifiers and high-cardinality location fields
- Prepared feature matrix (X) and target (y)

6️⃣ Train–Test Split
- 80/20 split
- Stratified sampling to preserve churn distribution

7️⃣ Baseline Model
- Logistic Regression
- One-hot encoding applied after train–test split
- No hyperparameter tuning (baseline only)

8️⃣ Model Evaluation
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score
- Initial results showed good accuracy but low recall for churners, indicating a conservative model.

9️⃣ Decision Threshold Optimization
Default threshold (0.5) missed many churners
Lowered threshold to 0.3

#### Result:
1. Higher recall for churn
2. Fewer missed churners
3. Acceptable increase in false positives
4. This trade-off aligns with business priorities.

🔟 Final Model Summary
- The final model prioritizes churn detection over accuracy
- Designed as a decision-support tool, not an automated decision-maker
- Interpretable, practical, and business-aligned

### 📈 Key Insights
1. Customers with low tenure churn more
2. Month-to-month contracts show higher churn
3. Higher monthly charges are associated with churn
4. Fiber optic customers have higher churn risk
5. Threshold choice matters more than model complexity

⚠️ Limitations
1. No hyperparameter tuning
2. No advanced models (intentionally)
3. Assumes historical patterns continue
4. Designed for decision support, not automation

### 🧩 Tools & Libraries
1. Python
2. pandas, numpy
3. matplotlib
4. scikit-learn

### ✅ Conclusion
This project demonstrates a complete, honest, and industry-aligned machine learning workflow — from problem framing to business-aware model decisions — using a simple and interpretable baseline model.

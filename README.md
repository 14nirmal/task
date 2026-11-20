# Financial Fraud Detection & Risk Scoring Platform

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)]()

A production-grade, end-to-end Machine Learning and Risk Scoring system that detects potentially fraudulent financial transactions in real time. Built strictly according to industry risk compliance standards, handling extreme class imbalance (~0.39% fraud prevalence) with zero data leakage, calibrated operational decision thresholds, and a high-performance **FastAPI** REST interface.

---

## 1. System Architecture

```text
fraud-risk-platform/
│
├── data/
│   ├── raw/                      <- Raw Kaggle transaction datasets (fraudTrain.csv, fraudTest.csv)
│   └── processed/                <- Stratified held-out test set (test_set.csv)
│
├── notebooks/
│   ├── eda_analysis.py           <- Automated EDA analysis runner
│   ├── eda_notebook.ipynb        <- Interactive Task 1 EDA notebook
│   ├── preprocessing_pipeline.ipynb <- Interactive Task 2 Preprocessing notebook
│   └── model_development_evaluation.ipynb <- Interactive Task 3 & 4 Model evaluation notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          <- Leak-free Scikit-Learn preprocessing & feature engineering
│   ├── train.py                  <- Multi-model training, tuning & comparison pipeline
│   └── evaluate.py               <- Quantitative test evaluation & threshold optimization
│
├── models/
│   ├── fraud_model.joblib        <- Champion production pipeline (Random Forest)
│   ├── model_comparison_results.csv
│   └── *.joblib                  <- Persisted candidate model pipelines
│
├── reports/
│   ├── evaluation_report.md      <- Full evaluation report
│   └── figures/                  <- Generated plots (imbalance, amounts, hours, PR/ROC curves)
│
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py          <- Unit tests for feature engineering & transformers
│   └── test_api.py               <- Integration tests for FastAPI endpoints, schemas & errors
│
├── app.py                        <- Production FastAPI REST application
├── requirements.txt              <- Python dependencies
└── README.md                     <- Comprehensive project documentation
```

---

## 2. Dataset Mapping & Sparkov / Kaggle Specification

The project uses the benchmark **Credit Card Transactions Fraud Detection Dataset** by Kartik Shenoy (`kartik2112/fraud-detection`).

| Assessment Specification | Kaggle Column | Processing & Engineering |
| :--- | :--- | :--- |
| **TransactionID** | `trans_num` | Excluded from modeling to prevent identity memorization & data leakage |
| **Amount** | `amt` | Scaled via `RobustScaler` and compressed via `log1p` |
| **TransactionTime** | `trans_date_trans_time` | Extracted `transaction_hour` (0–23) |
| **MerchantCategory** | `category` | Encoded with `OneHotEncoder(handle_unknown='ignore')` |
| **CustomerAge** | `dob` | Derived `customer_age_days = (trans_date - dob).days` |
| **FailedAttempts** | `previous_declines_24h` | Rolling decline velocity in preceding 24-hour window |
| **IsFraud (Target)** | `is_fraud` | Binary label: `0` = Legitimate (99.61%), `1` = Fraud (0.39%) |

---

## 3. Preprocessing & Feature Engineering Decisions

1. **Leak-Free Pipeline Architecture:**
   - Preprocessing is encapsulated inside a Scikit-Learn `Pipeline` and `ColumnTransformer`.
   - `pipeline.fit()` is called strictly on the training partition (`X_train`). Test data (`X_test`) and incoming API payloads are strictly transformed without re-fitting.
2. **Domain-Driven Derived Features:**
   - **`is_night_window`:** Flags transactions occurring between 22:00 and 04:00 (EDA discovered a >5x fraud spike during off-peak hours).
   - **`amount_to_age_ratio`:** $\frac{\text{amount}}{\text{customer\_age\_days} + 1}$, capturing account takeover and rapid spending on new accounts.
   - **`declines_risk_interaction`:** $\text{previous\_declines\_24h} \times \ln(1 + \text{amount})$, detecting card-probing attacks.
   - **`log_amount`:** Normalizes extreme positive skew in transaction amounts.
3. **Handling Class Imbalance:**
   - Employs **Cost-Sensitive Learning (`class_weight='balanced_subsample'`)**.
   - Unlike synthetic oversampling (SMOTE), this generates zero artificial data artifacts, avoids memory blow-up, and guarantees **zero test-set resampling contamination** (Assessment Rule 6).

---

## 4. Model Comparison & Evaluation Results

Models were evaluated on **10,000 unseen test transactions** preserving the true operational fraud prevalence (~0.39%):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Average Precision) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **99.72%** | **62.79%** | **69.23%** | **0.6585** | **0.9625** | **0.7106** |
| **HistGradientBoosting** | 98.97% | 23.77% | 74.36% | 0.3602 | 0.9552 | 0.5347 |
| **Logistic Regression (Baseline)** | 96.17% | 8.65% | 92.31% | 0.1582 | 0.9461 | 0.3802 |

### **Why Random Forest was Selected**
- Achieves the highest **PR-AUC (0.7106)**, significantly outperforming linear and gradient-boosted baselines.
- Effectively balances fraud capture (**Recall = 69.2%**) with false alarm mitigation (**Precision = 62.8%**).
- Resilient to financial outliers and automatically models non-linear interactions across merchant categories and transaction hours.

### **Why Accuracy is Misleading**
Because legitimate transactions comprise **99.61%** of all volume, a trivial model predicting "Legitimate" on all transactions achieves **99.61% accuracy** while missing **100% of fraud attacks**. Evaluation must be governed by **PR-AUC**, **Recall**, and **Precision**.

---

## 5. Operational Decision Thresholds & Multi-Tier Risk Bands

Rather than arbitrary binary classification ($p = 0.50$), the platform applies calibrated decision boundaries based on financial cost-benefit optimization:

| Fraud Probability ($p$) | Risk Band (`risk_band`) | Operational Decision (`decision`) | System Action |
| :---: | :---: | :---: | :--- |
| **$p < 0.20$** | **`LOW`** | **`APPROVE`** | Instant approval, frictionless checkout for >98% of clean traffic. |
| **$0.20 \le p < 0.65$** | **`MEDIUM`** | **`REVIEW`** | Trigger step-up verification (SMS OTP / 3DS challenge) or fraud queue. |
| **$p \ge 0.65$** | **`HIGH`** | **`DECLINE`** | Instant block and card freeze to halt balance drainage. |

---

## 6. REST API Documentation (FastAPI)

### **Run the API Server**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Interactive Swagger UI documentation is automatically available at:
`http://localhost:8000/docs`

---

### **Endpoint A: `GET /health`**
**Response (HTTP 200):**
```json
{
  "status": "HEALTHY",
  "model_loaded": true,
  "model_path": "models/fraud_model.joblib",
  "model_name": "Random_Forest_Champion_Pipeline",
  "version": "1.0.0",
  "uptime_seconds": 45.2,
  "thresholds": {
    "APPROVE_LOW_RISK": "p < 0.2",
    "REVIEW_MEDIUM_RISK": "0.2 <= p < 0.65",
    "DECLINE_HIGH_RISK": "p >= 0.65"
  }
}
```

---

### **Endpoint B: `POST /predict`**
**Request Body:**
```json
{
  "amount": 249.50,
  "merchant_category": "shopping_net",
  "transaction_hour": 2,
  "customer_age_days": 180,
  "previous_declines_24h": 3
}
```
**Response (HTTP 200):**
```json
{
  "fraud_probability": 0.6883,
  "risk_band": "HIGH",
  "decision": "DECLINE"
}
```

---

### **Endpoint C: `POST /predict-batch`**
**Request Body:**
```json
[
  {
    "amount": 15.00,
    "merchant_category": "grocery_pos",
    "transaction_hour": 14,
    "customer_age_days": 1200,
    "previous_declines_24h": 0
  },
  {
    "amount": 1250.00,
    "merchant_category": "shopping_net",
    "transaction_hour": 3,
    "customer_age_days": 45,
    "previous_declines_24h": 4
  }
]
```
**Response (HTTP 200):**
```json
{
  "total_transactions": 2,
  "high_risk_count": 1,
  "medium_risk_count": 0,
  "low_risk_count": 1,
  "predictions": [
    {
      "index": 0,
      "fraud_probability": 0.0116,
      "risk_band": "LOW",
      "decision": "APPROVE"
    },
    {
      "index": 1,
      "fraud_probability": 0.7090,
      "risk_band": "HIGH",
      "decision": "DECLINE"
    }
  ]
}
```

---

## 7. Installation, Execution & Testing Guide

### **1. Environment Setup**
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd fraud-risk-platform
python -m pip install -r requirements.txt
```

### **2. Run Exploratory Data Analysis (EDA)**
```bash
python notebooks/eda_analysis.py
```
*Outputs visualizations to `reports/figures/`.*

### **3. Run Model Training & Comparison**
```bash
python src/train.py
```
*Trains baseline and ensemble models, selects the champion, and saves `models/fraud_model.joblib`.*

### **4. Run Model Evaluation & Threshold Analysis**
```bash
python src/evaluate.py
```
*Generates test metric tables, PR/ROC curves, and `reports/evaluation_report.md`.*

### **5. Run Automated Pytest Suite**
```bash
python -m pytest tests/ -v
```
*Runs 12 unit and integration tests covering pipeline transformations, API responses, schema validation, and restart simulation.*

### **6. Start the FastAPI Production Server**
```bash
python app.py
```
*Navigate to `http://localhost:8000/docs` to test endpoints interactively.*


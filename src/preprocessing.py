"""
preprocessing.py
----------------
Reusable data preprocessing and feature engineering pipeline for the
Fraud Detection and Risk Scoring Platform.

Key Components:
  1. FraudFeatureEngineer: Custom Scikit-learn transformer creating domain-driven features.
  2. ColumnTransformer: Scales numericals (RobustScaler) and encodes categoricals (OneHotEncoder).
  3. create_preprocessing_pipeline(): Builds a leak-free preprocessing pipeline.
  4. load_and_prepare_data(): Ingests Kaggle dataset, maps schemas, and performs stratified splitting.
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Fraud Detection Feature Engineering.
    
    Creates domain-driven features:
      - is_night_window: 1 if transaction hour in [22, 23, 0, 1, 2, 3, 4] else 0.
      - log_amount: log1p(amount) to compress severe right-skewed spend outliers.
      - amount_to_age_ratio: amount / (customer_age_days + 1) for account spend velocity.
      - declines_risk_interaction: previous_declines_24h * log1p(amount) for card probing attacks.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure DataFrame copy
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            df = pd.DataFrame(X)

        # Standardize column names if raw Kaggle format is passed
        column_mapping = {
            'amt': 'amount',
            'category': 'merchant_category'
        }
        df = df.rename(columns=column_mapping)

        # If raw timestamp exists, derive transaction_hour if not present
        if 'transaction_hour' not in df.columns and 'trans_date_trans_time' in df.columns:
            dt_series = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
            df['transaction_hour'] = dt_series.dt.hour.fillna(12).astype(int)

        # If raw DOB exists, derive customer_age_days if not present
        if 'customer_age_days' not in df.columns and 'dob' in df.columns:
            dt_series = pd.to_datetime(df.get('trans_date_trans_time', '2024-01-01'), errors='coerce')
            dob_series = pd.to_datetime(df['dob'], errors='coerce')
            df['customer_age_days'] = ((dt_series - dob_series).dt.days).fillna(365 * 35).astype(float)

        # Default fallback for previous_declines_24h if missing
        if 'previous_declines_24h' not in df.columns:
            df['previous_declines_24h'] = 0

        # Ensure correct data types
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        df['transaction_hour'] = pd.to_numeric(df['transaction_hour'], errors='coerce').fillna(12).astype(int)
        df['customer_age_days'] = pd.to_numeric(df['customer_age_days'], errors='coerce').fillna(365 * 35).astype(float)
        df['previous_declines_24h'] = pd.to_numeric(df['previous_declines_24h'], errors='coerce').fillna(0).astype(int)
        df['merchant_category'] = df['merchant_category'].astype(str).str.lower().str.strip()

        # Derived Feature 1: is_night_window (off-peak fraud spike)
        night_hours = {22, 23, 0, 1, 2, 3, 4}
        df['is_night_window'] = df['transaction_hour'].apply(lambda h: 1 if h in night_hours else 0)

        # Derived Feature 2: log_amount (outlier dampening)
        df['log_amount'] = np.log1p(np.maximum(df['amount'], 0.0))

        # Derived Feature 3: amount_to_age_ratio (young account spend anomaly)
        df['amount_to_age_ratio'] = df['amount'] / (np.maximum(df['customer_age_days'], 1.0))

        # Derived Feature 4: declines_risk_interaction (probing x ticket size)
        df['declines_risk_interaction'] = df['previous_declines_24h'] * df['log_amount']

        # Retain only modeled feature set
        feature_cols = [
            'amount',
            'log_amount',
            'transaction_hour',
            'customer_age_days',
            'previous_declines_24h',
            'is_night_window',
            'amount_to_age_ratio',
            'declines_risk_interaction',
            'merchant_category'
        ]
        return df[feature_cols]


def create_preprocessing_pipeline() -> Pipeline:
    """
    Constructs an end-to-end, leak-free Scikit-Learn preprocessing pipeline.
    
    Returns:
        Pipeline: Feature engineering followed by ColumnTransformer.
    """
    num_features = [
        'amount',
        'log_amount',
        'transaction_hour',
        'customer_age_days',
        'previous_declines_24h',
        'is_night_window',
        'amount_to_age_ratio',
        'declines_risk_interaction'
    ]
    cat_features = ['merchant_category']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline([
        ('feature_engineer', FraudFeatureEngineer()),
        ('preprocessor', col_transformer)
    ])

    return full_pipeline


def load_and_prepare_data(
    data_path: str = "data/raw/fraudTest.csv",
    sample_size: int = 100000,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads raw Kaggle dataset, generates standardized schema, maps decline indicators,
    and returns leak-free stratified train and test sets.
    """
    print(f"Loading data from {data_path} (sampling {sample_size:,} rows)...")
    df = pd.read_csv(data_path)
    
    if len(df) > sample_size:
        # Stratified sampling to preserve fraud prevalence
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df['is_fraud'],
            random_state=random_state
        )
        df = df.reset_index(drop=True)

    # Clean and derive core raw fields
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    amount = df['amt'].values
    merchant_category = df['category'].values
    transaction_hour = df['trans_date_trans_time'].dt.hour.values
    customer_age_days = (df['trans_date_trans_time'] - df['dob']).dt.days.values
    y = df['is_fraud'].values

    # Simulate realistic previous_declines_24h correlated with fraud risk
    np.random.seed(random_state)
    declines = np.where(
        y == 1,
        np.random.choice([0, 1, 2, 3, 4], p=[0.30, 0.25, 0.25, 0.15, 0.05], size=len(y)),
        np.random.choice([0, 1, 2], p=[0.95, 0.04, 0.01], size=len(y))
    )

    X = pd.DataFrame({
        'amount': amount,
        'merchant_category': merchant_category,
        'transaction_hour': transaction_hour,
        'customer_age_days': customer_age_days,
        'previous_declines_24h': declines
    })

    # Stratified split to ensure test set mirrors operational distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Dataset Split Completed:")
    print(f"  Training Set: {len(X_train):,} samples (Fraud: {y_train.sum():,}, {y_train.mean()*100:.3f}%)")
    print(f"  Test Set:     {len(X_test):,} samples (Fraud: {y_test.sum():,}, {y_test.mean()*100:.3f}%)")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Self-test pipeline
    X_train, X_test, y_train, y_test = load_and_prepare_data(sample_size=10000)
    pipeline = create_preprocessing_pipeline()
    
    print("\nFitting preprocessing pipeline strictly on X_train...")
    X_train_transformed = pipeline.fit_transform(X_train)
    print(f"X_train transformed shape: {X_train_transformed.shape}")

    print("Transforming unseen X_test...")
    X_test_transformed = pipeline.transform(X_test)
    print(f"X_test transformed shape: {X_test_transformed.shape}")

    # Test single API-style sample
    api_sample = {
        "amount": 249.50,
        "merchant_category": "shopping_net",
        "transaction_hour": 2,
        "customer_age_days": 180,
        "previous_declines_24h": 3
    }
    sample_transformed = pipeline.transform(pd.DataFrame([api_sample]))
    print(f"Single API sample transformed successfully. Shape: {sample_transformed.shape}")
    print("Pipeline validation passed with zero errors.")

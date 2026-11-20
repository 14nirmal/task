"""
test_pipeline.py
----------------
Unit tests verifying the preprocessing and feature engineering pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import create_preprocessing_pipeline, FraudFeatureEngineer, load_and_prepare_data


def test_feature_engineer_derived_features():
    """Verify that custom feature engineering creates all required derived columns."""
    sample_data = pd.DataFrame([{
        "amount": 250.0,
        "merchant_category": "shopping_net",
        "transaction_hour": 2,
        "customer_age_days": 180,
        "previous_declines_24h": 3
    }])
    
    engineer = FraudFeatureEngineer()
    df_transformed = engineer.transform(sample_data)
    
    # Check night window: 2:00 AM should be flagged as 1
    assert "is_night_window" in df_transformed.columns
    assert df_transformed["is_night_window"].iloc[0] == 1
    
    # Check log_amount: log1p(250) ~ 5.525
    assert "log_amount" in df_transformed.columns
    assert np.isclose(df_transformed["log_amount"].iloc[0], np.log1p(250.0))
    
    # Check amount_to_age_ratio: 250 / 180
    assert "amount_to_age_ratio" in df_transformed.columns
    assert np.isclose(df_transformed["amount_to_age_ratio"].iloc[0], 250.0 / 180.0)
    
    # Check declines interaction: 3 * log1p(250)
    assert "declines_risk_interaction" in df_transformed.columns
    assert np.isclose(df_transformed["declines_risk_interaction"].iloc[0], 3 * np.log1p(250.0))


def test_pipeline_transform_shape():
    """Test full ColumnTransformer fit and transform without leakage."""
    train_data = pd.DataFrame([
        {"amount": 10.0, "merchant_category": "grocery_pos", "transaction_hour": 12, "customer_age_days": 500, "previous_declines_24h": 0},
        {"amount": 500.0, "merchant_category": "shopping_net", "transaction_hour": 1, "customer_age_days": 100, "previous_declines_24h": 2},
        {"amount": 45.0, "merchant_category": "gas_transport", "transaction_hour": 18, "customer_age_days": 1000, "previous_declines_24h": 0}
    ])
    
    pipeline = create_preprocessing_pipeline()
    transformed_train = pipeline.fit_transform(train_data)
    
    assert transformed_train.shape[0] == 3
    assert transformed_train.shape[1] > 5

    # Test unknown category in test set (should not raise error)
    test_data = pd.DataFrame([{
        "amount": 120.0,
        "merchant_category": "novel_unknown_category",
        "transaction_hour": 14,
        "customer_age_days": 300,
        "previous_declines_24h": 1
    }])
    transformed_test = pipeline.transform(test_data)
    assert transformed_test.shape[0] == 1
    assert transformed_test.shape[1] == transformed_train.shape[1]

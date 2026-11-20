"""
test_api.py
-----------
Comprehensive automated test suite for the FastAPI Fraud Detection REST API.

Tests:
  1. test_health_endpoint: verifies system status, model name, and threshold rules.
  2. test_predict_valid_transaction: verifies single transaction scoring and schema.
  3. test_predict_negative_amount_fails: verifies validation on negative amounts.
  4. test_predict_missing_required_fields: verifies validation on missing attributes.
  5. test_predict_invalid_hour_fails: verifies hour boundary checks (0-23).
  6. test_predict_batch_valid: verifies multi-transaction batch processing and aggregation.
  7. test_predict_batch_empty_fails: verifies empty batch returns HTTP 400.
  8. test_predict_batch_dict_envelope: verifies {"transactions": [...]} envelope support.
  9. test_risk_threshold_boundaries: verifies threshold cutoffs (APPROVE, REVIEW, DECLINE).
  10. test_api_restart_simulation: verifies model persistence and re-loading after restart.
"""

import pytest
import os
from pydantic import ValidationError
from fastapi import HTTPException
from app import (
    health,
    predict,
    predict_batch,
    load_model,
    assign_risk_and_decision,
    TransactionInput,
    PredictionOutput,
    BatchPredictionOutput,
    HealthResponse,
    APPROVE_THRESHOLD,
    DECLINE_THRESHOLD
)


@pytest.fixture(autouse=True)
def ensure_model_loaded():
    """Ensure the model pipeline is loaded before running tests."""
    load_model("models/fraud_model.joblib")


def test_health_endpoint():
    """Test 1: Verify health endpoint returns healthy status and model metadata."""
    res = health()
    assert isinstance(res, HealthResponse)
    assert res.status == "HEALTHY"
    assert res.model_loaded is True
    assert "Random_Forest" in res.model_name
    assert "APPROVE_LOW_RISK" in res.thresholds


def test_predict_valid_transaction():
    """Test 2: Verify valid single transaction returns probability, risk band, and decision."""
    tx = TransactionInput(
        amount=249.50,
        merchant_category="shopping_net",
        transaction_hour=2,
        customer_age_days=180,
        previous_declines_24h=3
    )
    res = predict(tx)
    assert isinstance(res, PredictionOutput)
    assert 0.0 <= res.fraud_probability <= 1.0
    assert res.risk_band in ["LOW", "MEDIUM", "HIGH"]
    assert res.decision in ["APPROVE", "REVIEW", "DECLINE"]


def test_predict_negative_amount_fails():
    """Test 3: Verify negative transaction amount raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        TransactionInput(
            amount=-50.0,
            merchant_category="grocery_pos",
            transaction_hour=14,
            customer_age_days=500,
            previous_declines_24h=0
        )
    assert "greater than 0" in str(exc_info.value)


def test_predict_missing_required_fields():
    """Test 4: Verify missing required fields raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        TransactionInput.model_validate({
            "amount": 100.0,
            "merchant_category": "shopping_net"
            # Missing transaction_hour and customer_age_days
        })
    errors = str(exc_info.value)
    assert "transaction_hour" in errors
    assert "customer_age_days" in errors


def test_predict_invalid_hour_fails():
    """Test 5: Verify hour outside 0-23 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        TransactionInput(
            amount=50.0,
            merchant_category="shopping_pos",
            transaction_hour=25, # Invalid hour
            customer_age_days=300,
            previous_declines_24h=0
        )
    assert "less than or equal to 23" in str(exc_info.value)


def test_predict_batch_valid():
    """Test 6: Verify valid batch prediction returns predictions for all transactions."""
    batch_data = [
        TransactionInput(
            amount=15.00,
            merchant_category="grocery_pos",
            transaction_hour=14,
            customer_age_days=1200,
            previous_declines_24h=0
        ),
        TransactionInput(
            amount=1450.00,
            merchant_category="shopping_net",
            transaction_hour=3,
            customer_age_days=30,
            previous_declines_24h=5
        )
    ]
    res = predict_batch(batch_data)
    assert isinstance(res, BatchPredictionOutput)
    assert res.total_transactions == 2
    assert len(res.predictions) == 2
    assert res.predictions[0].index == 0
    assert res.predictions[1].index == 1
    assert res.high_risk_count + res.medium_risk_count + res.low_risk_count == 2


def test_predict_batch_empty_fails():
    """Test 7: Verify empty batch request raises HTTP 400 Bad Request."""
    with pytest.raises(HTTPException) as exc_info:
        predict_batch([])
    assert exc_info.value.status_code == 400
    assert "Batch request cannot be empty" in exc_info.value.detail


def test_predict_batch_dict_envelope():
    """Test 8: Verify {"transactions": [...]} envelope format is supported."""
    envelope = {
        "transactions": [
            {
                "amount": 35.0,
                "merchant_category": "food_dining",
                "transaction_hour": 19,
                "customer_age_days": 400,
                "previous_declines_24h": 0
            }
        ]
    }
    res = predict_batch(envelope)
    assert res.total_transactions == 1
    assert res.predictions[0].decision in ["APPROVE", "REVIEW", "DECLINE"]


def test_risk_threshold_boundaries():
    """Test 9: Verify mathematical correctness of operational decision bands."""
    # Low risk band (p < 0.20)
    p_low, band, dec = assign_risk_and_decision(0.08)
    assert band == "LOW" and dec == "APPROVE"

    # Medium risk band (0.20 <= p < 0.65)
    p_med, band, dec = assign_risk_and_decision(0.42)
    assert band == "MEDIUM" and dec == "REVIEW"

    # High risk band (p >= 0.65)
    p_high, band, dec = assign_risk_and_decision(0.78)
    assert band == "HIGH" and dec == "DECLINE"


def test_api_restart_simulation():
    """Test 10: Confirm that the API works after reloading model pipeline into memory."""
    import app as app_module
    
    # Simulate restart by resetting model pipeline
    app_module.model_pipeline = None
    reloaded_pipeline = load_model("models/fraud_model.joblib")
    assert reloaded_pipeline is not None
    
    # Test scoring immediately after restart
    tx = TransactionInput(
        amount=55.0,
        merchant_category="gas_transport",
        transaction_hour=10,
        customer_age_days=900,
        previous_declines_24h=0
    )
    res = predict(tx)
    assert res.decision == "APPROVE"
    assert res.risk_band == "LOW"

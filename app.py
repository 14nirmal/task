"""
app.py
------
Production FastAPI REST API for the Fraud Detection and Risk Scoring Platform.

Endpoints:
  - GET  /health         : System health, uptime, and model status.
  - POST /predict        : Real-time fraud scoring for a single transaction.
  - POST /predict-batch  : Batch fraud scoring with array validation.
  - GET  /docs           : Interactive Swagger UI API documentation.
"""

import os
import sys
import time

# Ensure repository root is in sys.path
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import joblib
import pandas as pd
from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------------------------------
# Operational Decision Threshold Rules
# --------------------------------------------------------------------------
APPROVE_THRESHOLD = 0.20   # p < 0.20 -> LOW risk, APPROVE
DECLINE_THRESHOLD = 0.65   # p >= 0.65 -> HIGH risk, DECLINE
                           # 0.20 <= p < 0.65 -> MEDIUM risk, REVIEW

def assign_risk_and_decision(probability: float):
    """Map calibrated fraud probability to operational risk band and action decision."""
    prob_rounded = round(float(probability), 4)
    if prob_rounded < APPROVE_THRESHOLD:
        return prob_rounded, "LOW", "APPROVE"
    elif prob_rounded < DECLINE_THRESHOLD:
        return prob_rounded, "MEDIUM", "REVIEW"
    else:
        return prob_rounded, "HIGH", "DECLINE"


# --------------------------------------------------------------------------
# Pydantic Schemas for Strict Input Validation
# --------------------------------------------------------------------------
class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD (must be strictly positive)")
    merchant_category: str = Field(..., min_length=1, description="Merchant category or type")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    customer_age_days: float = Field(..., ge=0, description="Customer account age in days")
    previous_declines_24h: int = Field(0, ge=0, description="Declined transaction count in past 24 hours")

    @field_validator("merchant_category")
    @classmethod
    def clean_category(cls, v: str) -> str:
        v_clean = str(v).strip().lower()
        if not v_clean:
            raise ValueError("merchant_category cannot be blank")
        return v_clean


class PredictionOutput(BaseModel):
    fraud_probability: float = Field(..., description="Estimated probability of fraud [0.0 - 1.0]")
    risk_band: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    decision: str = Field(..., description="Action decision: APPROVE, REVIEW, or DECLINE")


class BatchPredictionItem(PredictionOutput):
    index: int = Field(..., description="Index of the transaction in the batch request")


class BatchPredictionOutput(BaseModel):
    total_transactions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_name: str
    version: str
    uptime_seconds: float
    thresholds: dict


# --------------------------------------------------------------------------
# Model Loading & Application Initialization
# --------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.joblib")
model_pipeline = None
start_time = time.time()

def load_model(path: str = MODEL_PATH):
    """Loads the serialized model pipeline into memory once at startup."""
    global model_pipeline
    if model_pipeline is None:
        if os.path.exists(path):
            print(f"Loading champion model from {path}...")
            model_pipeline = joblib.load(path)
            print("Model successfully loaded into memory.")
        else:
            print(f"Warning: Model not found at {path}!")
    return model_pipeline

# Ingest model into memory at startup
load_model()

app = FastAPI(
    title="Fraud Detection & Risk Scoring Platform API",
    description="Production-grade Financial Fraud Detection REST API powered by Machine Learning and FastAPI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom Validation Exception Handler to return HTTP 400 Bad Request
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append({"field": field, "message": msg, "type": err.get("type")})

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Input validation error",
            "details": errors
        }
    )


# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Returns service health status, model availability, and operational configuration."""
    pipe = load_model()
    is_ready = pipe is not None
    uptime = round(time.time() - start_time, 2)
    
    return HealthResponse(
        status="HEALTHY" if is_ready else "UNHEALTHY",
        model_loaded=is_ready,
        model_path=MODEL_PATH,
        model_name="Random_Forest_Champion_Pipeline",
        version="1.0.0",
        uptime_seconds=uptime,
        thresholds={
            "APPROVE_LOW_RISK": f"p < {APPROVE_THRESHOLD}",
            "REVIEW_MEDIUM_RISK": f"{APPROVE_THRESHOLD} <= p < {DECLINE_THRESHOLD}",
            "DECLINE_HIGH_RISK": f"p >= {DECLINE_THRESHOLD}"
        }
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(transaction: TransactionInput):
    """
    Evaluates a single financial transaction.
    Returns: fraud_probability, risk_band (LOW/MEDIUM/HIGH), and decision (APPROVE/REVIEW/DECLINE).
    """
    pipe = load_model()
    if pipe is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable or service is initializing."
        )

    # Convert to single-row DataFrame matching the preprocessing pipeline
    df_input = pd.DataFrame([transaction.model_dump()])

    try:
        prob = pipe.predict_proba(df_input)[0, 1]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

    prob_clean, risk_band, decision = assign_risk_and_decision(prob)

    return PredictionOutput(
        fraud_probability=prob_clean,
        risk_band=risk_band,
        decision=decision
    )


@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["Prediction"])
def predict_batch(transactions: Union[List[TransactionInput], Dict[str, Any]]):
    """
    Evaluates a batch of financial transactions.
    Supports either a direct JSON array `[...]` or an object `{"transactions": [...]}`.
    """
    pipe = load_model()
    if pipe is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable or service is initializing."
        )

    # Handle dictionary envelope or direct list
    if isinstance(transactions, dict):
        items = transactions.get("transactions", [])
    elif isinstance(transactions, list):
        items = transactions
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed batch format. Expected array or object with 'transactions' key."
        )

    if not items or len(items) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch request cannot be empty. Please provide at least one valid transaction."
        )

    # Validate items
    validated_items = []
    for idx, raw_tx in enumerate(items):
        if isinstance(raw_tx, TransactionInput):
            validated_items.append(raw_tx.model_dump())
        elif isinstance(raw_tx, dict):
            try:
                tx = TransactionInput(**raw_tx)
                validated_items.append(tx.model_dump())
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Validation failed for item at index {idx}: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Item at index {idx} must be a valid transaction object."
            )

    df_batch = pd.DataFrame(validated_items)

    try:
        probabilities = pipe.predict_proba(df_batch)[:, 1]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )

    predictions = []
    high_count, med_count, low_count = 0, 0, 0

    for idx, p in enumerate(probabilities):
        prob_clean, band, dec = assign_risk_and_decision(p)
        if band == "HIGH":
            high_count += 1
        elif band == "MEDIUM":
            med_count += 1
        else:
            low_count += 1

        predictions.append(BatchPredictionItem(
            index=idx,
            fraud_probability=prob_clean,
            risk_band=band,
            decision=dec
        ))

    return BatchPredictionOutput(
        total_transactions=len(predictions),
        high_risk_count=high_count,
        medium_risk_count=med_count,
        low_risk_count=low_count,
        predictions=predictions
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

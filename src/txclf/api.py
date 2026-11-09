"""FastAPI service: POST /predict (batch), GET /health.

Run: uvicorn txclf.api:app --port 8000   (model path from TXCLF_MODEL)
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from txclf import __version__
from txclf.model import TxModel, load_model


class PredictRequest(BaseModel):
    descriptions: list[str] = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(3, ge=1, le=10)


class Candidate(BaseModel):
    category: str
    probability: float


class Prediction(BaseModel):
    description: str
    category: str
    probability: float
    top_k: list[Candidate]
    known_terms: int


class PredictResponse(BaseModel):
    predictions: list[Prediction]


@lru_cache(maxsize=1)
def get_model() -> TxModel:
    return load_model(os.environ.get("TXCLF_MODEL", "models/model.joblib"))


app = FastAPI(title="txclf", version=__version__, description="Transaction category prediction")


@app.get("/health")
def health() -> dict:
    try:
        model = get_model()
    except Exception as exc:  # report, don't crash the probe
        return {"status": "error", "detail": str(exc)}
    return {
        "status": "ok",
        "version": __version__,
        "categories": len(model.categories),
        "trained_at": model.metadata.get("trained_at"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        model = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model not loaded: {exc}") from exc
    return PredictResponse(predictions=model.predict(req.descriptions, top_k=req.top_k))

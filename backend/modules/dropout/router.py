"""
modules/dropout/router.py – Dropout risk prediction endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.auth.router import get_current_user
from backend.modules.dropout.model import predict_dropout, train
from backend.modules.mlflow_tracker import log_run

router = APIRouter(prefix="/dropout", tags=["Dropout Risk"])


class StudentFeatures(BaseModel):
    age: int = 17
    studytime: int = 2
    failures: int = 0
    absences: int = 5
    G1: float = 12.0
    G2: float = 11.0
    sex: str = "M"
    address: str = "U"
    schoolsup: str = "no"
    famsup: str = "yes"
    freetime: int = 3
    goout: int = 3
    health: int = 3
    famrel: int = 4
    model_type: str = "rf"


@router.post("/predict")
async def predict(req: StudentFeatures, user=Depends(get_current_user)):
    try:
        features = req.model_dump()
        model_type = features.pop("model_type", "rf")
        result = predict_dropout(features, model_type)
        log_run("dropout_inference",
                params={"model": model_type, "user": user.get("email")},
                metrics={"dropout_prob": result["dropout_probability"]})
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/retrain")
async def retrain(user=Depends(get_current_user)):
    if user.get("role") not in ("teacher", "admin"):
        raise HTTPException(403, "Teachers only")
    try:
        metrics = train()
        return {"status": "retrained", **metrics}
    except Exception as e:
        raise HTTPException(500, str(e))

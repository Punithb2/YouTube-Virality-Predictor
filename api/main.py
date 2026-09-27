import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="YouTube Virality Predictor API",
    description="Predicts expected 30-day video views using LightGBM and metadata features.",
    version="1.0.0"
)

# Load serialized model and metadata on startup
MODEL_PATH = "models/lightgbm_model.pkl"
META_PATH = "models/inference_metadata.json"

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

class VideoPredictionRequest(BaseModel):
    channel_id: str = Field(..., example="UC_x5XG1OV2P6uZZ5FSM9Ttw")
    title: str = Field(..., example="I Survived 100 Days in Hardcore Minecraft!")
    description: Optional[str] = Field("", example="In this video, we build an iron farm...")
    tags: Optional[List[str]] = Field(default=[], example=["minecraft", "hardcore", "survival"])
    duration_seconds: int = Field(..., example=1200)
    publish_hour: Optional[int] = Field(15, ge=0, le=23, description="Hour of upload (0-23)")
    is_weekend: Optional[int] = Field(0, ge=0, le=1, description="1 if published on Saturday/Sunday")

class VideoPredictionResponse(BaseModel):
    predicted_views_30_days: int
    baseline_source: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=VideoPredictionResponse)
def predict_views(payload: VideoPredictionRequest):
    # Resolve baseline channel history
    if payload.channel_id in meta["channel_stats"]:
        ch_avg = meta["channel_stats"][payload.channel_id]
        baseline_source = "channel_history"
    else:
        ch_avg = meta["global_avg_log_views"]
        baseline_source = "global_fallback"

    # Engineer features exactly as the model expects
    input_data = {
        "channel_avg_log_views": ch_avg,
        "log_age": meta["fixed_log_age"],  # Locked at 30 days
        "title_length": len(payload.title),
        "description_length": len(payload.description or ""),
        "tag_count": len(payload.tags or []),
        "duration_seconds": payload.duration_seconds,
        "publish_hour": payload.publish_hour,
        "is_weekend": payload.is_weekend
    }

    # Order dataframe strictly by training features
    df_input = pd.DataFrame([input_data])[meta["features"]]

    # Predict and back-transform from log space
    log_pred = float(model.predict(df_input)[0])
    raw_pred = int(np.expm1(log_pred))

    return VideoPredictionResponse(
        predicted_views_30_days=max(0, raw_pred),
        baseline_source=baseline_source
    )
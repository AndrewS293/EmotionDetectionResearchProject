from pydantic import BaseModel
from typing import Dict, List, Any


class SensorWindow(BaseModel):

    data: List[List[float]]


class PredictionResponse(BaseModel):

    prediction: str

    confidence: float

    model_predictions: Dict[str, Any]

    model_probabilities: Dict[str, Any]

    reasoning: Dict[str, Any]
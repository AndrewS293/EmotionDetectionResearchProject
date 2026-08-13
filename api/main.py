from fastapi import FastAPI
import numpy as np

from api.schemas import SensorWindow
from inference.model_loader import load_models
from inference.predictor import predict
from llm.ollama_reasoner import OllamaReasoner
from inference.ensemble import weighted_ensemble
from inference.schema import create_report


WEIGHTS = {
    "XGBoost": 0.965,
    
    "Random Forest": 0.865,
    
    "Gradient Boosting": 0.70,
    
    "SVM": 0.41,

    "Logistic Regression": 0.14,
    
    "KNN": 0.32
}


app = FastAPI(
    title="MindSense API",
    version="1.0.0"
)


models, feature_names = load_models()

@app.post("/predict")
def predict_emotion(request: SensorWindow):

    sensor_window = np.array(
        request.data,
        dtype=float
    )

    prediction_result = predict(
        sensor_window
    )

    ensemble_result = weighted_ensemble(
        prediction_result["predictions"],
        WEIGHTS
    )

    prediction_report = create_report(
        prediction_result,
        ensemble_result
    )

    reasoning = OllamaReasoner().analyze_state(
        prediction_report
    )

    return {
        "prediction": ensemble_result["prediction"],
        "confidence": ensemble_result["confidence"],
        "models": prediction_result,
        "ensemble": ensemble_result,
        "reasoning": reasoning
    }



@app.get("/health")
def health():

    return {
        "status": "ok",
        "models_loaded": len(models),
        "ollama": OllamaReasoner().test_connection()
    }

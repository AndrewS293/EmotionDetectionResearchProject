import joblib
from pathlib import Path


MODEL_PATH = Path("models")


def load_models():

    models = {

        "XGBoost":
            joblib.load(
                MODEL_PATH / "xgboost.pkl"
            ),

        "Random Forest":
            joblib.load(
                MODEL_PATH / "randomforest.pkl"
            ),

        "SVM":
            joblib.load(
                MODEL_PATH / "svm.pkl"
            ),

        "Logistic Regression":
            joblib.load(
                MODEL_PATH / "logistic.pkl"
            ),

        "KNN":
            joblib.load(
                MODEL_PATH / "knn.pkl"
            )
    }


    scaler = joblib.load(
        MODEL_PATH / "scaler.pkl"
    )


    encoder = joblib.load(
        MODEL_PATH / "label_encoder.pkl"
    )


    return models, scaler, encoder
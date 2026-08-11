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
                MODEL_PATH / "random_forest.pkl"
            ),

        "SVM":
            joblib.load(
                MODEL_PATH / "svm.pkl"
            ),

        "Logistic Regression":
            joblib.load(
                MODEL_PATH / "logistic_regression.pkl"
            ),

        "KNN":
            joblib.load(
                MODEL_PATH / "knn.pkl"
            ),

        "Gradient Boosting":
            joblib.load(
                MODEL_PATH / "gradient_boosting.pkl"
            )
    }


    return models

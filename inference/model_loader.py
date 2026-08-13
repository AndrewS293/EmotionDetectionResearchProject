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

    feature_names = {
        "XGBoost":
            joblib.load(
                MODEL_PATH / "xgboost_features.pkl"
            ),
        "Random Forest":
            joblib.load(
                MODEL_PATH / "random_forest_features.pkl"
            ),
        "SVM":
            joblib.load(
                MODEL_PATH / "svm_features.pkl"
            ),
        "Logistic Regression":
            joblib.load(
                MODEL_PATH / "logistic_regression_features.pkl"
            ),
        "KNN":
            joblib.load(
                MODEL_PATH / "knn_features.pkl"
            ),
        "Gradient Boosting":
            joblib.load(
                MODEL_PATH / "gradient_boosting_features.pkl"
            )
    }

    return models, feature_names

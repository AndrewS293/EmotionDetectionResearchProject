# ============================================================
# MODEL PREDICTOR
# ============================================================

import numpy as np

from inference.model_loader import load_models

from pipelines.feature_extraction import (
    build_logistic_features,
    build_svm_features,
    build_knn_features,
    build_rf_features,
    build_gb_features,
    build_xgb_features
)


# ============================================================
# LOAD MODELS
# ============================================================

models, feature_names = load_models()


# ============================================================
# CLASS LABELS
# ============================================================

CLASS_NAMES = {

    0: "Baseline",

    1: "Stress",

    2: "Amusement",

    3: "Meditation"
}


# ============================================================
# MODEL-SPECIFIC FEATURE BUILDERS
# ============================================================

FEATURE_BUILDERS = {

    "Logistic Regression":
        build_logistic_features,

    "SVM":
        build_svm_features,

    "KNN":
        build_knn_features,

    "Random Forest":
        build_rf_features,

    "Gradient Boosting":
        build_gb_features,

    "XGBoost":
        build_xgb_features
}


# ============================================================
# PREDICT
# ============================================================

def predict(sensor_window):

    predictions = {}

    probabilities = {}

    model_features = {}


    # ========================================================
    # RUN EACH MODEL
    # ========================================================

    for name, model in models.items():

        #print(
        #    f"Running {name}..."
        #)


        # ----------------------------------------------------
        # Get model-specific feature builder
        # ----------------------------------------------------

        builder = FEATURE_BUILDERS[name]


        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features, extracted_names = (
            builder(sensor_window)
        )


        # ----------------------------------------------------
        # Convert to model input
        # ----------------------------------------------------

        X = np.asarray(
            features,
            dtype=float
        ).reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # Store feature values
        # ----------------------------------------------------

        model_features[name] = {

            feature_name: float(value)

            for feature_name, value
            in zip(
                extracted_names,
                features
            )
        }


        # ----------------------------------------------------
        # Predict
        #
        # IMPORTANT:
        #
        # No manual scaling here.
        #
        # Logistic Regression, SVM, and KNN contain their
        # StandardScaler inside their saved Pipeline.
        # ----------------------------------------------------

        pred = model.predict(
            X
        )[0]


        # ----------------------------------------------------
        # Convert class number to readable label
        # ----------------------------------------------------

        predictions[name] = CLASS_NAMES.get(
            int(pred),
            str(pred)
        )


        # ----------------------------------------------------
        # Probability prediction
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            proba = model.predict_proba(
                X
            )[0]


            probabilities[name] = {

                CLASS_NAMES.get(
                    int(class_id),
                    str(class_id)
                ): float(probability)

                for class_id, probability
                in zip(
                    model.classes_,
                    proba
                )
            }

        else:

            probabilities[name] = None


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "predictions":
            predictions,

        "probabilities":
            probabilities,

        "features":
            model_features
    }

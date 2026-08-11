# ============================================================
# TRAIN + EVALUATE MODELS
# ============================================================
#
# Purpose:
#   Research/evaluation pipeline.
#
# This script:
#   1. Loads WESAD
#   2. Builds model-specific features
#   3. Performs an 80/20 stratified split
#   4. Trains each model
#   5. Evaluates each model
#   6. Stores predictions, probabilities, metrics,
#      and feature information
#
# IMPORTANT:
#   This file is for MODEL EVALUATION.
#
#   The production .pkl files are created separately by:
#
#       export_final_models.py
#
# ============================================================

import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# DATA LOADER
# ============================================================

from pipelines.data_loader import (
    load_all_subjects as load_all_wesad
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

from pipelines.feature_extraction import (
    create_model_windows,

    build_logistic_features,
    build_svm_features,
    build_knn_features,
    build_rf_features,
    build_gb_features,
    build_xgb_features
)


# ============================================================
# EVALUATION
# ============================================================

from pipelines.evaluate import evaluate_model


# ============================================================
# MODEL BUILDERS
# ============================================================

from pipelines.logistics_pipeline import (
    build_model as logistic_model
)

from pipelines.svm_pipeline import (
    build_model as svm_model
)

from pipelines.randomforest_pipeline import (
    build_model as rf_model
)

from pipelines.xgboost_pipeline import (
    build_model as xgb_model
)

from pipelines.knn_pipeline import (
    build_model as knn_model
)

from pipelines.gradientboost_pipeline import (
    build_model as gb_model
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# FEATURE BUILDERS
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
# MODEL BUILDERS
# ============================================================

MODEL_BUILDERS = {

    "Logistic Regression":
        logistic_model,

    "SVM":
        svm_model,

    "KNN":
        knn_model,

    "Random Forest":
        rf_model,

    "Gradient Boosting":
        gb_model,

    "XGBoost":
        xgb_model
}


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_models(DATA_DIR):

    print(
        "\n=========================================="
    )

    print(
        "LOADING WESAD DATA"
    )

    print(
        "==========================================\n"
    )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    signals, labels = load_all_wesad(
        DATA_DIR
    )


    print(
        f"Loaded {len(signals)} subjects."
    )


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    models = {}

    results = {}

    X_models = {}

    y_models = {}

    model_feature_names = {}

    test_data = {}

    predictions = {}

    probabilities = {}


    # ========================================================
    # TRAIN EACH MODEL
    # ========================================================

    for name, model_builder in MODEL_BUILDERS.items():

        print(
            "\n=========================================="
        )

        print(
            f"TRAINING {name}"
        )

        print(
            "=========================================="
        )


        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        model = model_builder()


        # ----------------------------------------------------
        # Select feature builder
        # ----------------------------------------------------

        builder = FEATURE_BUILDERS[name]


        all_X = []

        all_y = []

        feature_names = None


        # ----------------------------------------------------
        # Build windows/features
        # ----------------------------------------------------

        for i, (signal, label) in enumerate(
            zip(signals, labels)
        ):

            print(
                f"Processing subject "
                f"{i + 1}/{len(signals)}",
                end="\r"
            )


            Xi, yi, current_feature_names = (
                create_model_windows(
                    signal,
                    label,
                    builder
                )
            )


            all_X.extend(Xi)

            all_y.extend(yi)


            # Feature names should be identical
            # for every subject.

            if feature_names is None:

                feature_names = (
                    current_feature_names
                )


        print()


        # ----------------------------------------------------
        # Convert to NumPy arrays
        # ----------------------------------------------------

        X = np.asarray(all_X)

        y = np.asarray(all_y)


        print(
            f"Dataset shape: {X.shape}"
        )

        print(
            f"Labels shape: {y.shape}"
        )


        # ----------------------------------------------------
        # Store complete dataset
        # ----------------------------------------------------

        X_models[name] = X

        y_models[name] = y

        model_feature_names[name] = (
            feature_names
        )


        # ----------------------------------------------------
        # Train/test split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )
        )


        test_data[name] = {

            "X_test": X_test,

            "y_test": y_test
        }


        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        print(
            f"Training {name}..."
        )


        model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        predictions[name] = y_pred


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities[name] = (
                model.predict_proba(
                    X_test
                )
            )

        else:

            probabilities[name] = None


        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            name
        )


        results[name] = metrics

        models[name] = model


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"\n{name} Results:"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"F1: "
            f"{metrics['f1']:.4f}"
        )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n\n=========================================="
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "==========================================\n"
    )


    for name, metrics in results.items():

        print(
            f"{name:<22}"
            f"| Accuracy: "
            f"{metrics['accuracy']:.4f}"
            f" | F1: "
            f"{metrics['f1']:.4f}"
        )


    print(
        "\n=========================================="
    )


    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return {

        "models": models,

        "signals": signals,

        "labels": labels,

        "X_models": X_models,

        "y_models": y_models,

        "model_feature_names":
            model_feature_names,

        "test_data":
            test_data,

        "predictions":
            predictions,

        "probabilities":
            probabilities,

        "results":
            results
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    DATA_DIR = r"data\WESAD"


    train_models(
        DATA_DIR
    )
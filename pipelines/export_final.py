# ============================================================
# EXPORT FINAL PRODUCTION MODELS
# ============================================================
#
# Purpose:
#   Train the final production versions of the models and
#   save them as .pkl files.
#
# These models are NOT evaluated here.
#
# They are intended for the deployed inference pipeline.
#
#
# Training:
#
#   WESAD
#      ↓
#   Model-specific feature extraction
#      ↓
#   Full approved training dataset
#      ↓
#   Model.fit()
#      ↓
#   Save fitted model
#
#
# The resulting .pkl files will later be loaded by:
#
#       inference/model_loader.py
#
#
# IMPORTANT:
#
# Logistic Regression, SVM, and KNN already contain
# StandardScaler in their build_model() functions.
#
# Therefore the scaler is saved INSIDE their .pkl files.
#
# Random Forest, Gradient Boosting, and XGBoost do not
# require a scaler.
#
# ============================================================

import os
import json
import joblib
import numpy as np


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

MODEL_DIR = "models"

DATASET_NAME = "WESAD"


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


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
# MODEL FILE NAMES
# ============================================================

MODEL_FILENAMES = {

    "Logistic Regression":
        "logistic_regression.pkl",

    "SVM":
        "svm.pkl",

    "KNN":
        "knn.pkl",

    "Random Forest":
        "random_forest.pkl",

    "Gradient Boosting":
        "gradient_boosting.pkl",

    "XGBoost":
        "xgboost.pkl"
}


# ============================================================
# EXPORT FUNCTION
# ============================================================

def export_final_models(
    DATA_DIR
):

    print(
        "\n=========================================="
    )

    print(
        "FINAL MODEL EXPORT"
    )

    print(
        "==========================================\n"
    )


    # ========================================================
    # LOAD WESAD
    # ========================================================

    print(
        "Loading WESAD..."
    )


    signals, labels = load_all_wesad(
        DATA_DIR
    )


    print(
        f"Loaded {len(signals)} subjects."
    )


    # ========================================================
    # STORAGE
    # ========================================================

    feature_names_by_model = {}

    metadata = {

        "dataset":
            DATASET_NAME,

        "models": {}
    }


    # ========================================================
    # TRAIN EACH FINAL MODEL
    # ========================================================

    for name, model_builder in MODEL_BUILDERS.items():

        print(
            "\n=========================================="
        )

        print(
            f"EXPORTING {name}"
        )

        print(
            "=========================================="
        )


        # ----------------------------------------------------
        # Create fresh model
        # ----------------------------------------------------

        model = model_builder()


        # ----------------------------------------------------
        # Select model-specific feature builder
        # ----------------------------------------------------

        builder = FEATURE_BUILDERS[name]


        all_X = []

        all_y = []

        feature_names = None


        # ----------------------------------------------------
        # Build features for every subject
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


            # Save feature names once

            if feature_names is None:

                feature_names = (
                    current_feature_names
                )


        print()


        # ----------------------------------------------------
        # Convert to NumPy arrays
        # ----------------------------------------------------

        X = np.asarray(
            all_X
        )

        y = np.asarray(
            all_y
        )


        print(
            f"Dataset shape: {X.shape}"
        )

        print(
            f"Labels shape: {y.shape}"
        )


        # ----------------------------------------------------
        # Store feature names
        # ----------------------------------------------------

        feature_names_by_model[name] = (
            feature_names
        )


        # ----------------------------------------------------
        # Train on ALL approved WESAD data
        # ----------------------------------------------------

        print(
            f"\nTraining final {name}..."
        )


        model.fit(
            X,
            y
        )


        # ----------------------------------------------------
        # Save trained model
        # ----------------------------------------------------

        filename = MODEL_FILENAMES[name]


        model_path = os.path.join(
            MODEL_DIR,
            filename
        )


        joblib.dump(
            model,
            model_path
        )


        print(
            f"Saved model:"
            f"\n  {model_path}"
        )


        # ----------------------------------------------------
        # Save model-specific feature names
        # ----------------------------------------------------

        feature_filename = (
            filename[:-4]
            + "_features.pkl"
        )


        feature_path = os.path.join(
            MODEL_DIR,
            feature_filename
        )


        joblib.dump(
            feature_names,
            feature_path
        )


        print(
            f"Saved features:"
            f"\n  {feature_path}"
        )


        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        metadata["models"][name] = {

            "model_file":
                filename,

            "feature_file":
                feature_filename,

            "number_of_samples":
                int(X.shape[0]),

            "number_of_features":
                int(X.shape[1]),

            "features":
                list(feature_names)
        }


    # ========================================================
    # SAVE ALL FEATURE INFORMATION
    # ========================================================

    all_features_path = os.path.join(
        MODEL_DIR,
        "model_features.pkl"
    )


    joblib.dump(
        feature_names_by_model,
        all_features_path
    )


    print(
        "\nSaved all feature information:"
        f"\n  {all_features_path}"
    )


    # ========================================================
    # SAVE TRAINING METADATA
    # ========================================================

    metadata_path = os.path.join(
        MODEL_DIR,
        "training_metadata.json"
    )


    with open(
        metadata_path,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


    print(
        "\nSaved training metadata:"
        f"\n  {metadata_path}"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n\n=========================================="
    )

    print(
        "FINAL MODELS EXPORTED"
    )

    print(
        "==========================================\n"
    )


    for name, filename in MODEL_FILENAMES.items():

        print(
            f"{name:<22}"
            f" → {filename}"
        )


    print(
        "\n=========================================="
    )

    print(
        "EXPORT COMPLETE"
    )

    print(
        "==========================================\n"
    )


    return metadata


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHANGE THIS TO YOUR WESAD DIRECTORY
    # --------------------------------------------------------

    DATA_DIR = r"data\WESAD"


    export_final_models(
        DATA_DIR
    )
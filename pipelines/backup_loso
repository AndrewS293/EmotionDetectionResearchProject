import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from pipelines.data_loader import load_subject
from pipelines.feature_extraction import (
    create_model_windows,
    build_logistic_features,
    build_svm_features,
    build_knn_features,
    build_rf_features,
    build_gb_features,
    build_xgb_features
)

from pipelines.logistics_pipeline import build_model as logistic_model
from pipelines.svm_pipeline import build_model as svm_model
from pipelines.randomforest_pipeline import build_model as rf_model
from pipelines.xgboost_pipeline import build_model as xgb_model
from pipelines.knn_pipeline import build_model as knn_model
from pipelines.gradientboost_pipeline import build_model as gb_model


# ============================================================
# CONFIGURATION
# ============================================================

EMOTION_LABELS = [
    "Baseline",
    "Stress",
    "Amusement",
    "Meditation"
]


# These are the same weights you have been using
WEIGHTS = {

    "XGBoost": 0.199,

    "Random Forest": 0.177,

    "Gradient Boosting": 0.20,

    "SVM": 0.149,

    "Logistic Regression": 0.134,

    "KNN": 0.141
}


# ============================================================
# MODEL DEFINITIONS
# ============================================================

FEATURE_BUILDERS = {
    "Logistic Regression": build_logistic_features,
    "SVM": build_svm_features,
    "KNN": build_knn_features,
    "Random Forest": build_rf_features,
    "Gradient Boosting": build_gb_features,
    "XGBoost": build_xgb_features
}


MODEL_BUILDERS = {
    "Logistic Regression": logistic_model,
    "SVM": svm_model,
    "KNN": knn_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model
}


# ============================================================
# LOAD SUBJECTS
# ============================================================

def load_wesad_subjects(data_dir):

    subjects = []

    for filename in sorted(os.listdir(data_dir)):

        if not filename.endswith(".pkl"):
            continue

        path = os.path.join(data_dir, filename)

        print(f"Loading {filename}")

        signal, labels = load_subject(path)

        subjects.append({
            "name": filename.replace(".pkl", ""),
            "signal": signal,
            "labels": labels
        })

    return subjects


# ============================================================
# CREATE WINDOWS FOR ONE SUBJECT
# ============================================================

def create_subject_features(subject):

    signal = subject["signal"]
    labels = subject["labels"]

    subject_features = {}
    subject_labels = {}

    for model_name, builder in FEATURE_BUILDERS.items():

        X, y, feature_names = create_model_windows(
            signal,
            labels,
            builder
        )

        subject_features[model_name] = X
        subject_labels[model_name] = y

    return subject_features, subject_labels


# ============================================================
# ENSEMBLE
# ============================================================

def weighted_ensemble(models, X_test):

    probability_predictions = {}
    weighted_probabilities = None

    total_weight = sum(WEIGHTS.values())

    for name, model in models.items():

        if not hasattr(model, "predict_proba"):
            continue

        probabilities = model.predict_proba(
            X_test[name]
        )

        probability_predictions[name] = probabilities

        weight = WEIGHTS[name]

        weighted = probabilities * weight

        if weighted_probabilities is None:
            weighted_probabilities = weighted
        else:
            weighted_probabilities += weighted

    weighted_probabilities /= total_weight

    predictions = np.argmax(
        weighted_probabilities,
        axis=1
    )

    return predictions, weighted_probabilities


# ============================================================
# LOSO EVALUATION
# ============================================================

def run_loso(data_dir):

    print("\n")
    print("=" * 70)
    print("LEAVE-ONE-SUBJECT-OUT (LOSO) EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD SUBJECTS
    # --------------------------------------------------------

    subjects = load_wesad_subjects(data_dir)

    print(f"\nLoaded {len(subjects)} subjects.")

    # --------------------------------------------------------
    # PRE-COMPUTE FEATURES
    #
    # This does NOT train anything.
    # We are simply converting each subject into feature windows.
    # --------------------------------------------------------

    processed_subjects = []

    for i, subject in enumerate(subjects):

        print(
            f"\nCreating features for "
            f"{subject['name']} ({i + 1}/{len(subjects)})"
        )

        X, y = create_subject_features(subject)

        processed_subjects.append({
            "name": subject["name"],
            "X": X,
            "y": y
        })

        for model_name in X:
            print(
                f"  {model_name}: "
                f"{X[model_name].shape}"
            )

    # --------------------------------------------------------
    # STORAGE FOR GLOBAL RESULTS
    # --------------------------------------------------------

    all_true = []
    all_predictions = {
        name: []
        for name in MODEL_BUILDERS
    }

    all_ensemble_predictions = []

    subject_results = []

    # ========================================================
    # LOSO LOOP
    # ========================================================

    for test_index in range(len(processed_subjects)):

        test_subject = processed_subjects[test_index]

        train_subjects = [
            subject
            for i, subject in enumerate(processed_subjects)
            if i != test_index
        ]

        print("\n")
        print("=" * 70)
        print(
            f"TEST SUBJECT: {test_subject['name']}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # TRAIN / TEST DATA FOR EACH MODEL
        # ----------------------------------------------------

        X_train = {}
        y_train = {}

        X_test = {}
        y_test = {}

        for model_name in MODEL_BUILDERS:

            train_X = []
            train_y = []

            for subject in train_subjects:

                train_X.extend(
                    subject["X"][model_name]
                )

                train_y.extend(
                    subject["y"][model_name]
                )

            X_train[model_name] = np.asarray(train_X)
            y_train[model_name] = np.asarray(train_y)

            X_test[model_name] = np.asarray(
                test_subject["X"][model_name]
            )

            y_test[model_name] = np.asarray(
                test_subject["y"][model_name]
            )

        # ----------------------------------------------------
        # TRAIN EACH MODEL
        # ----------------------------------------------------

        models = {}

        subject_model_predictions = {}

        for model_name, model_builder in MODEL_BUILDERS.items():

            print(
                f"\nTraining {model_name}"
            )

            model = model_builder()

            model.fit(
                X_train[model_name],
                y_train[model_name]
            )

            models[model_name] = model

            predictions = model.predict(
                X_test[model_name]
            )

            subject_model_predictions[
                model_name
            ] = predictions

            all_predictions[
                model_name
            ].extend(predictions)

            accuracy = accuracy_score(
                y_test[model_name],
                predictions
            )

            f1 = f1_score(
                y_test[model_name],
                predictions,
                average="macro"
            )

            print(
                f"  Accuracy: {accuracy:.4f}"
            )

            print(
                f"  Macro F1: {f1:.4f}"
            )

        # ----------------------------------------------------
        # ENSEMBLE
        # ----------------------------------------------------

        ensemble_pred, ensemble_probs = weighted_ensemble(
            models,
            X_test
        )

        ensemble_true = y_test["XGBoost"]

        ensemble_accuracy = accuracy_score(
            ensemble_true,
            ensemble_pred
        )

        ensemble_f1 = f1_score(
            ensemble_true,
            ensemble_pred,
            average="macro"
        )

        print("\nENSEMBLE")
        print(
            f"  Accuracy: {ensemble_accuracy:.4f}"
        )

        print(
            f"  Macro F1: {ensemble_f1:.4f}"
        )

        # ----------------------------------------------------
        # STORE GLOBAL RESULTS
        # ----------------------------------------------------

        all_true.extend(ensemble_true)

        all_ensemble_predictions.extend(
            ensemble_pred
        )

        subject_results.append({
            "subject": test_subject["name"],
            "accuracy": ensemble_accuracy,
            "f1": ensemble_f1
        })

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    all_true = np.asarray(all_true)
    all_ensemble_predictions = np.asarray(
        all_ensemble_predictions
    )

    print("\n")
    print("=" * 70)
    print("FINAL LOSO RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # INDIVIDUAL MODELS
    # --------------------------------------------------------

    print("\nINDIVIDUAL MODELS")

    for model_name in MODEL_BUILDERS:

        predictions = np.asarray(
            all_predictions[model_name]
        )

        accuracy = accuracy_score(
            all_true,
            predictions
        )

        f1 = f1_score(
            all_true,
            predictions,
            average="macro"
        )

        print(
            f"{model_name:22s} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {f1:.4f}"
        )

    # --------------------------------------------------------
    # ENSEMBLE
    # --------------------------------------------------------

    ensemble_accuracy = accuracy_score(
        all_true,
        all_ensemble_predictions
    )

    ensemble_f1 = f1_score(
        all_true,
        all_ensemble_predictions,
        average="macro"
    )

    print("\nENSEMBLE")

    print(
        f"Accuracy: {ensemble_accuracy:.4f}"
    )

    print(
        f"Macro F1: {ensemble_f1:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            all_true,
            all_ensemble_predictions,
            target_names=EMOTION_LABELS,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nCONFUSION MATRIX")

    cm = confusion_matrix(
        all_true,
        all_ensemble_predictions
    )

    print(cm)

    # --------------------------------------------------------
    # PER-SUBJECT RESULTS
    # --------------------------------------------------------

    print("\nPER-SUBJECT ENSEMBLE RESULTS")

    for result in subject_results:

        print(
            f"{result['subject']:10s} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro F1: {result['f1']:.4f}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LOSO SUMMARY")
    print("=" * 70)

    print(
        f"Subjects tested: {len(subjects)}"
    )

    print(
        f"Total test windows: {len(all_true)}"
    )

    print(
        f"Ensemble Accuracy: {ensemble_accuracy:.4f}"
    )

    print(
        f"Ensemble Macro F1: {ensemble_f1:.4f}"
    )

    return {
        "subjects": subject_results,
        "y_true": all_true,
        "ensemble_predictions": all_ensemble_predictions,
        "individual_predictions": all_predictions,
        "ensemble_accuracy": ensemble_accuracy,
        "ensemble_macro_f1": ensemble_f1,
        "confusion_matrix": cm
    }
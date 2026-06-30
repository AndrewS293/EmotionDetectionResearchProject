from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


EMOTION_LABELS = [
    'Baseline',
    'Stress',
    'Amusement',
    'Meditation'
]


def evaluate_model(model, X_test, y_test, name):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")

    # CLASSIFICATION REPORT WITH REAL LABEL NAMES
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=EMOTION_LABELS
        )
    )

    # CONFUSION MATRIX
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=EMOTION_LABELS
    )

    fig, ax = plt.subplots(figsize=(7,6))

    disp.plot(
        ax=ax,
        cmap='Blues',
        values_format='d'
    )

    plt.title(f"{name} Confusion Matrix")

    plt.show()

    return {
        "accuracy": acc,
        "f1": f1
    }
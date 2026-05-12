import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

#evaluates models by getting accuracy and f1 for now, prints out confusion matrix
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

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title(name)

    plt.show()

    return {
        "accuracy": acc,
        "f1": f1
    }
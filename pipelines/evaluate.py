from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


EMOTION_LABELS = [
    'Happy',
    'Sad',
    'Calm',
    'Angry'
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
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_LABELS)
    fig, ax = plt.subplots(figsize=(7,6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    plt.title(f"{name} Confusion Matrix")
    
    # Place a text box in the bottom right corner of the figure
    # transform=ax.transAxes uses a 0-1 scale for the plotting area
    fig.text(
        0.75, 0.02, # Coordinates: X=75% across figure, Y=2% from bottom
        f"Accuracy: {acc:.2%}", 
        fontsize=12, fontweight='bold', color='black',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    # Use tight_layout so the text box doesn't overlap labels
    plt.tight_layout()
    plt.show()



    return {
        "accuracy": acc,
        "f1": f1
    }
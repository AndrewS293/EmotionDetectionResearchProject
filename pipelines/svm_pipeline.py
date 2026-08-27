from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_model():
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf',
            C=2.0,
            gamma = 0.05,
            class_weight='balanced',
            probability=True
        ))
    ])

    return model 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_model():
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C = 0.2, max_iter=3000,class_weight='balanced'))
    ])

    return model
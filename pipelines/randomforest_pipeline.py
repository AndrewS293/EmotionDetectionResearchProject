from sklearn.ensemble import RandomForestClassifier


def build_model():
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced'
    )

    return model

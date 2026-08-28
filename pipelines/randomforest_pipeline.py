from sklearn.ensemble import RandomForestClassifier


def build_model():
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=None,
        class_weight='balanced'
    )

    return model

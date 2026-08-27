from sklearn.ensemble import RandomForestClassifier


def build_model():
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=4,
        max_features='log2',
        class_weight='balanced'
    )

    return model

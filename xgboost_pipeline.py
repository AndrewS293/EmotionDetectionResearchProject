from xgboost import XGBClassifier


def build_model():
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=4,
        eval_metric='mlogloss'
    )

    return model

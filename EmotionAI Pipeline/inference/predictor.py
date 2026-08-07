from inference.model_loader import load_models
from pipelines.feature_extraction import extract_features


models, scaler, encoder = load_models()



def predict(sensor_window):


    # 1. Extract features

    features = extract_features(
        sensor_window
    )


    # 2. Convert to model format

    X = features.values.reshape(1,-1)


    # 3. Scale

    X_scaled = scaler.transform(X)



    predictions = {}

    probabilities = {}



    # 4. Run every model

    for name, model in models.items():


        pred = model.predict(
            X_scaled
        )[0]


        predictions[name] = encoder.inverse_transform(
            [pred]
        )[0]


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities[name] = (
                model.predict_proba(X_scaled)
            )


    return {

        "predictions": predictions,

        "probabilities": probabilities,

        "features": features.to_dict()

    }
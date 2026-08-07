def weighted_ensemble(
    predictions,
    weights
):

    scores = {}


    for model, emotion in predictions.items():


        weight = weights.get(
            model,
            1
        )


        scores[emotion] = (
            scores.get(emotion,0)
            +
            weight
        )


    final_prediction = max(
        scores,
        key=scores.get
    )


    confidence = (
        scores[final_prediction]
        /
        sum(scores.values())
    )


    return {

        "prediction":
            final_prediction,

        "confidence":
            confidence

    }
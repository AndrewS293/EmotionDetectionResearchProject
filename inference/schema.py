def create_report(
    results,
    ensemble
):


    return {

        "final_prediction":
            ensemble["prediction"],


        "confidence":
            ensemble["confidence"],


        "model_predictions":
            results["predictions"],


        "physiological_features":
            results["features"]

    }
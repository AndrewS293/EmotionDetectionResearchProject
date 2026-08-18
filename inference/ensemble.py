# ============================================================ 
# WEIGHTED PROBABILITY ENSEMBLE 
# ============================================================ 
import numpy as np 

CLASS_NAMES = [ "Baseline", "Stress", "Amusement", "Meditation" ] 

def weighted_ensemble( probabilities, weights ): 
    
    # -------------------------------------------------------- 
    # Initialize class scores 
    # -------------------------------------------------------- 
    
    scores = { emotion: 0.0 for emotion in CLASS_NAMES } 
    total_weight = 0.0 
    # ======================================================== 
    # COMBINE MODEL PROBABILITIES 
    # ======================================================== 
    for model_name, model_probs in probabilities.items(): 
    # # Skip models that don't provide probabilities 
        if model_probs is None: 
            continue 

        weight = weights.get( model_name, 0 ) # Skip zero-weight models 
        if weight <= 0: 
            continue 

        total_weight += weight 

        for emotion in CLASS_NAMES: 
            probability = model_probs.get( emotion, 0.0 ) 
            scores[emotion] += ( weight * probability ) 


    # ======================================================== 
    # SAFETY CHECK 
    # ======================================================== 
    if total_weight == 0: raise ValueError( "No valid model weights were provided." ) 

    # ======================================================== 
    # NORMALIZE 
    # ======================================================== 
    combined_probabilities = { emotion: score / total_weight for emotion, score in scores.items() } 
    # ======================================================== 
    # FINAL PREDICTION 
    # ======================================================== 
    final_prediction = max( combined_probabilities, key=combined_probabilities.get ) 
    # ======================================================== 
    # CONFIDENCE 
    # ======================================================== 
    confidence = combined_probabilities[ final_prediction ] 
    # ======================================================== 
    # RETURN 
    # ======================================================== 
    return { "prediction": final_prediction, "confidence": float(confidence), "probabilities": combined_probabilities }
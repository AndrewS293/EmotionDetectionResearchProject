import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------------------------------------
# Moderate manual weights
# ---------------------------------------------------------
# Start with these rather than aggressively balancing.
#
# 0 = Baseline
# 1 = Stress
# 2 = Amusement
# 3 = Meditation
#
CLASS_WEIGHTS = {
    0: 1.0,
    1: 1.2,
    2: 2.0,
    3: 2.5,
}


def get_sample_weights(y, method="manual"):
    """
    Return a sample-weight array for training.

    method:
        "manual"   -> moderate manually selected weights
        "balanced" -> sklearn's inverse-frequency balancing
        "none"     -> all samples have weight 1
    """

    y = np.asarray(y)

    if method == "none":
        return np.ones(len(y))

    if method == "balanced":
        return compute_sample_weight(
            class_weight="balanced",
            y=y
        )

    if method == "manual":
        return np.array([
            CLASS_WEIGHTS[int(label)]
            for label in y
        ])

    raise ValueError(
        f"Unknown balancing method: {method}"
    )
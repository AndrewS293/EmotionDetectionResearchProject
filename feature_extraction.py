import numpy as np
from scipy.stats import skew, kurtosis

#gets all our features
#takes data from a window and compresses it
def create_windows(signal, labels, window_size=200, step=50):
    X = []
    y = []

    for i in range(0, len(signal) - window_size, step):
        window = signal[i:i+window_size]
        label_window = labels[i:i+window_size]

        label = np.bincount(label_window).argmax()

        features = []

        for ch in range(window.shape[1]):
            w = window[:, ch]


            #gets our features - I'll add more later
            features.extend([
                np.mean(w),
                np.std(w),
                np.min(w),
                np.max(w),
                skew(w),
                kurtosis(w),
                np.median(w),
                np.percentile(w, 25),
                np.percentile(w, 75),
                np.sum(w**2)
            ])

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)



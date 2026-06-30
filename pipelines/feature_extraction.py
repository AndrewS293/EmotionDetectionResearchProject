# pipelines/feature_extraction.py

import numpy as np

from scipy.stats import (
    skew,
    kurtosis,
    entropy
)

from scipy.signal import find_peaks


# ============================================================
# SENSOR-SPECIFIC FEATURE EXTRACTORS
# ============================================================

# ------------------------------------------------------------
# EDA FEATURES
# ------------------------------------------------------------

def extract_eda_features(w):

    features = []

    feature_names = []

    # BASIC STATS
    features.extend([
        np.mean(w),
        np.std(w),
        np.min(w),
        np.max(w),
        np.median(w)
    ])

    feature_names.extend([
        'EDA Mean',
        'EDA Std',
        'EDA Min',
        'EDA Max',
        'EDA Median'
    ])

    # DERIVATIVES
    diff = np.diff(w)

    features.extend([
        np.mean(diff),
        np.std(diff)
    ])

    feature_names.extend([
        'EDA Diff Mean',
        'EDA Diff Std'
    ])

    # PEAK FEATURES
    peaks, _ = find_peaks(
        w,
        distance=5
    )

    peak_count = len(peaks)

    if peak_count > 0:

        peak_heights = w[peaks]

        avg_peak_height = np.mean(peak_heights)

    else:

        avg_peak_height = 0

    features.extend([
        peak_count,
        avg_peak_height
    ])

    feature_names.extend([
        'EDA Peak Count',
        'EDA Avg Peak Height'
    ])

  
    hist, _ = np.histogram(w, bins=20)

    features.append(
        entropy(hist + 1)
    )

    feature_names.append(
        'EDA Entropy'
    )

    return features, feature_names


# ------------------------------------------------------------
# BVP FEATURES
# ------------------------------------------------------------

def extract_bvp_features(w):

    features = []

    feature_names = []

    # BASIC STATS
    features.extend([
        np.mean(w),
        np.std(w),
        skew(w),
        kurtosis(w)
    ])

    feature_names.extend([
        'BVP Mean',
        'BVP Std',
        'BVP Skew',
        'BVP Kurtosis'
    ])

    # FFT FEATURES
    fft_vals = np.abs(
        np.fft.rfft(w)
    )

    features.extend([
        np.mean(fft_vals),
        np.std(fft_vals),
        np.max(fft_vals)
    ])

    feature_names.extend([
        'BVP FFT Mean',
        'BVP FFT Std',
        'BVP FFT Max'
    ])

    # PEAK FEATURES
    peaks, _ = find_peaks(
        w,
        distance=5
    )

    features.append(
        len(peaks)
    )

    feature_names.append(
        'BVP Peak Count'
    )

    # DERIVATIVES
    diff = np.diff(w)

    features.extend([
        np.mean(diff),
        np.std(diff)
    ])

    feature_names.extend([
        'BVP Diff Mean',
        'BVP Diff Std'
    ])

    return features, feature_names


# ------------------------------------------------------------
# TEMP FEATURES
# ------------------------------------------------------------

def extract_temp_features(w):

    features = []

    feature_names = []

    features.extend([
        np.mean(w),
        np.std(w),
        np.min(w),
        np.max(w)
    ])

    feature_names.extend([
        'TEMP Mean',
        'TEMP Std',
        'TEMP Min',
        'TEMP Max'
    ])

    diff = np.diff(w)

    features.extend([
        np.mean(diff),
        np.std(diff)
    ])

    feature_names.extend([
        'TEMP Diff Mean',
        'TEMP Diff Std'
    ])

    return features, feature_names


# ------------------------------------------------------------
# ACC FEATURES
# ------------------------------------------------------------

def extract_acc_features(
    acc_x,
    acc_y,
    acc_z
):

    features = []

    feature_names = []

    mag = np.sqrt(
        acc_x**2 +
        acc_y**2 +
        acc_z**2
    )

    features.extend([
        np.mean(mag),
        np.std(mag),
        np.max(mag),
        np.sum(mag**2)
    ])

    feature_names.extend([
        'ACC Magnitude Mean',
        'ACC Magnitude Std',
        'ACC Magnitude Max',
        'ACC Energy'
    ])

    fft_vals = np.abs(
        np.fft.rfft(mag)
    )

    features.extend([
        np.mean(fft_vals),
        np.max(fft_vals)
    ])

    feature_names.extend([
        'ACC FFT Mean',
        'ACC FFT Max'
    ])

    return features, feature_names


# ============================================================
# MODEL-SPECIFIC FEATURE BUILDERS
# ============================================================

# ------------------------------------------------------------
# LOGISTIC REGRESSION FEATURES
# ------------------------------------------------------------

def build_logistic_features(window):

    eda = window[:,0]
    bvp = window[:,1]
    temp = window[:,2]

    features = []
    feature_names = []

    for signal_name, w in zip(
        ['EDA', 'BVP', 'TEMP'],
        [eda, bvp, temp]
    ):

        features.extend([
            np.mean(w),
            np.std(w),
            np.min(w),
            np.max(w)
        ])

        feature_names.extend([
            f'{signal_name} Mean',
            f'{signal_name} Std',
            f'{signal_name} Min',
            f'{signal_name} Max'
        ])

    return np.nan_to_num(features), feature_names


# ------------------------------------------------------------
# SVM FEATURES
# ------------------------------------------------------------

def build_svm_features(window):

    eda = window[:,0]
    bvp = window[:,1]
    temp = window[:,2]

    features = []
    feature_names = []

    # EDA
    eda_features, eda_names = extract_eda_features(eda)

    features.extend(eda_features)
    feature_names.extend(eda_names)

    # BVP
    bvp_features, bvp_names = extract_bvp_features(bvp)

    features.extend(bvp_features)
    feature_names.extend(bvp_names)

    # TEMP
    temp_features, temp_names = extract_temp_features(temp)

    features.extend(temp_features)
    feature_names.extend(temp_names)

    return np.nan_to_num(features), feature_names


# ------------------------------------------------------------
# KNN FEATURES
# ------------------------------------------------------------

def build_knn_features(window):

    eda = window[:,0]
    bvp = window[:,1]

    features = []
    feature_names = []

    for signal_name, w in zip(
        ['EDA', 'BVP'],
        [eda, bvp]
    ):

        diff = np.diff(w)

        features.extend([
            np.mean(w),
            np.std(w),
            np.mean(diff)
        ])

        feature_names.extend([
            f'{signal_name} Mean',
            f'{signal_name} Std',
            f'{signal_name} Diff Mean'
        ])

    return np.nan_to_num(features), feature_names


# ------------------------------------------------------------
# RANDOM FOREST FEATURES
# ------------------------------------------------------------

def build_rf_features(window):

    eda = window[:,0]
    bvp = window[:,1]
    temp = window[:,2]

    acc_x = window[:,3]
    acc_y = window[:,4]
    acc_z = window[:,5]

    features = []
    feature_names = []

    eda_features, eda_names = extract_eda_features(eda)
    bvp_features, bvp_names = extract_bvp_features(bvp)
    temp_features, temp_names = extract_temp_features(temp)

    acc_features, acc_names = extract_acc_features(
        acc_x,
        acc_y,
        acc_z
    )

    features.extend(eda_features)
    features.extend(bvp_features)
    features.extend(temp_features)
    features.extend(acc_features)

    feature_names.extend(eda_names)
    feature_names.extend(bvp_names)
    feature_names.extend(temp_names)
    feature_names.extend(acc_names)

    return np.nan_to_num(features), feature_names


# ------------------------------------------------------------
# GRADIENT BOOSTING FEATURES
# ------------------------------------------------------------

def build_gb_features(window):

    return build_rf_features(window)


# ------------------------------------------------------------
# XGBOOST FEATURES
# ------------------------------------------------------------

def build_xgb_features(window):

    rf_features, rf_feature_names = build_rf_features(window)

    # Convert back to list so we can extend
    features = list(rf_features)

    feature_names = list(rf_feature_names)

    eda = window[:,0]
    bvp = window[:,1]

    extra_features = []
    extra_names = []

    for signal_name, w in zip(
        ['EDA', 'BVP'],
        [eda, bvp]
    ):

        extra_features.extend([
            skew(w),
            kurtosis(w)
        ])

        extra_names.extend([
            f'{signal_name} Global Skew',
            f'{signal_name} Global Kurtosis'
        ])

    features.extend(extra_features)

    feature_names.extend(extra_names)

    return np.nan_to_num(features), feature_names
# ============================================================
# WINDOW BUILDERS
# ============================================================

def create_model_windows(
    signal,
    labels,
    feature_builder,
    window_size=400,
    step=50
):

    X = []
    y = []

    for i in range(
        0,
        len(signal) - window_size,
        step
    ):

        window = signal[
            i:i+window_size
        ]

        label_window = labels[
            i:i+window_size
        ]

        label = np.bincount(
            label_window
        ).argmax()

        features, feature_names = feature_builder(window)

        X.append(features)

        y.append(label)

    return np.array(X), np.array(y), feature_names


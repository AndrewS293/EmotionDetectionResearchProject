import numpy as np


class SubjectBaseline:
    """
    Stores baseline statistics for a single subject.

    Baseline statistics are calculated separately for each subject.
    This allows features to describe changes relative to that
    subject's own physiological baseline.
    """

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, signals):
        """
        signals:
            Dictionary:
                {
                    "EDA": 1D array,
                    "BVP": 1D array,
                    "TEMP": 1D array
                }

        Only baseline data should be passed here.
        """

        for signal_name, values in signals.items():

            values = np.asarray(
                values,
                dtype=float
            )

            values = values[
                np.isfinite(values)
            ]

            if len(values) == 0:
                continue

            mean = np.mean(values)
            std = np.std(values)

            self.means[signal_name] = mean

            # Prevent division by zero
            self.stds[signal_name] = max(
                std,
                1e-8
            )

        return self

    def transform_summary(
        self,
        signal_name,
        values
    ):
        """
        Return subject-relative summary features.

        Returns:
            delta = current mean - baseline mean
            z     = delta / baseline std
            ratio = current mean / baseline mean
        """

        values = np.asarray(
            values,
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:

            return [
                0.0,
                0.0,
                1.0
            ]

        current_mean = np.mean(values)

        # No baseline available
        if signal_name not in self.means:

            return [
                current_mean,
                0.0,
                1.0
            ]

        baseline_mean = self.means[
            signal_name
        ]

        baseline_std = self.stds[
            signal_name
        ]

        delta = (
            current_mean -
            baseline_mean
        )

        z_score = (
            delta /
            baseline_std
        )

        if abs(baseline_mean) > 1e-8:

            ratio = (
                current_mean /
                baseline_mean
            )

        else:

            ratio = 1.0

        return [
            delta,
            z_score,
            ratio
        ]

    def feature_names(self, signal_name):

        return [
            f"{signal_name} Baseline Delta",
            f"{signal_name} Baseline Z",
            f"{signal_name} Baseline Ratio"
        ]


def build_subject_baseline(subject):
    """
    Build a baseline object from one WESAD subject.

    IMPORTANT:
    Only label 0 (Baseline) is used.

    This assumes subject['signal'] contains:
        column 0 = EDA
        column 1 = BVP
        column 2 = TEMP
        column 3 = ACC X
        column 4 = ACC Y
        column 5 = ACC Z

    and subject['labels'] contains the corresponding labels.
    """

    signal = subject["signal"]
    labels = subject["labels"]

    baseline_mask = (
        labels == 0
    )

    baseline_signals = {

        "EDA":
            signal[
                baseline_mask,
                0
            ],

        "BVP":
            signal[
                baseline_mask,
                1
            ],

        "TEMP":
            signal[
                baseline_mask,
                2
            ]
    }

    baseline = SubjectBaseline()

    baseline.fit(
        baseline_signals
    )

    return baseline
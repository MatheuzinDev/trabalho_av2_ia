import numpy as np


METRIC_KEYS = ("accuracy", "sensitivity", "specificity", "precision", "f1_score")
METRIC_LABELS = {
    "accuracy": "Acuracia",
    "sensitivity": "Sensibilidade",
    "specificity": "Especificidade",
    "precision": "Precisao",
    "f1_score": "F1-score",
}
METRIC_FILE_LABELS = {
    "accuracy": "acuracia",
    "sensitivity": "sensibilidade",
    "specificity": "especificidade",
    "precision": "precisao",
    "f1_score": "f1_score",
}


def calculate_validation_metrics(confusion_matrix):
    true_positives = confusion_matrix[0][0]
    true_negatives = confusion_matrix[1][1]
    false_positives = confusion_matrix[0][1]
    false_negatives = confusion_matrix[1][0]

    accuracy = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives
    )
    sensitivity = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) != 0
        else np.nan
    )
    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) != 0
        else np.nan
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) != 0
        else 0.0
    )
    f1_score = (
        (2 * precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) != 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score,
    }


def summarize_metric_values(values):
    metric_values = np.asarray(values, dtype=float)
    return {
        "mean": np.nanmean(metric_values),
        "std": np.nanstd(metric_values),
        "max": np.nanmax(metric_values),
        "min": np.nanmin(metric_values),
    }

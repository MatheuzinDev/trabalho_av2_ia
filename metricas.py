import numpy as np


def confusion_matrix_binary(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def compute_metrics(confusion_matrix):
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    total = confusion_matrix.sum()

    accuracy = safe_divide(tp + tn, total)
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    f1_score = safe_divide(2 * precision * sensitivity, precision + sensitivity)

    return {
        "acuracia": accuracy,
        "sensibilidade": sensitivity,
        "especificidade": specificity,
        "precisao": precision,
        "f1_score": f1_score,
    }

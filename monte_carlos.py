import numpy as np

from metricas import confusion_matrix_binary, compute_metrics


def _standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_test_std


def run_test(X, y, R, create_model, predict_model):
    valores = {
        "acuracia": [],
        "sensibilidade": [],
        "especificidade": [],
        "precisao": [],
        "f1_score": [],
    }

    for _ in range(R):
        indices = np.random.permutation(X.shape[0])
        test_size = int(np.floor(X.shape[0] * 0.2))

        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        X_train_std, X_test_std = _standardize_train_test(X_train, X_test)

        model = create_model(X_train_std, y_train)
        model.fit()

        y_pred = np.asarray(predict_model(model, X_test_std)).ravel()

        confusion_matrix = confusion_matrix_binary(y_test, y_pred)
        metrics = compute_metrics(confusion_matrix)

        for metric_name in valores:
            valores[metric_name].append(metrics[metric_name])

    resumo = {}

    for metric_name, metric_values in valores.items():
        metric_values = np.asarray(metric_values, dtype=float)
        resumo[metric_name] = {
            "media": float(np.mean(metric_values)),
            "desvio_padrao": float(np.std(metric_values)),
            "maior": float(np.max(metric_values)),
            "menor": float(np.min(metric_values)),
        }

    return {
        "valores": valores,
        "resumo": resumo,
    }

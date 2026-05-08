import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dados import create_train_test_split, normalize_train_test
from metricas import calculate_validation_metrics, summarize_metric_values
from modelos.perceptron import SimplePerceptron


class MonteCarloResults:
    def __init__(self, metric_keys):
        self.metric_keys = tuple(metric_keys)
        self.metrics = {metric_key: [] for metric_key in self.metric_keys}
        self.records = []

    def add_record(self, round_index, confusion_matrix, metric_values, learning_curve):
        record = {
            "round": round_index,
            "confusion_matrix": np.asarray(confusion_matrix, dtype=int),
            "metrics": metric_values,
            "learning_curve": list(learning_curve),
        }
        self.records.append(record)

        for metric_key in self.metric_keys:
            self.metrics[metric_key].append(metric_values[metric_key])

    def summary(self, metric_key):
        return summarize_metric_values(self.metrics[metric_key])

    def best_worst_cases(self, metric_key):
        values = np.asarray(self.metrics[metric_key], dtype=float)
        best_index = int(np.nanargmax(values))
        worst_index = int(np.nanargmin(values))
        return {
            "best": self.records[best_index],
            "worst": self.records[worst_index],
        }


class MonteCarloTester:
    def __init__(self, test_matrix, model_or_weights):
        self.test_matrix = test_matrix
        self.model_or_weights = model_or_weights

    def _bipolar_step_activation(self, x):
        if x >= 0:
            return 1
        return -1

    def run_test(self):
        X_test = self.test_matrix[:, :3]
        predictions = self._predict_batch(X_test)
        targets = self.test_matrix[:, -1].astype(int)

        true_positives = int(np.sum((targets == 1) & (predictions == 1)))
        true_negatives = int(np.sum((targets == -1) & (predictions == -1)))
        false_positives = int(np.sum((targets == -1) & (predictions == 1)))
        false_negatives = int(np.sum((targets == 1) & (predictions == -1)))

        return [[true_positives, false_positives], [false_negatives, true_negatives]]

    def _predict_batch(self, X_with_bias):
        if hasattr(self.model_or_weights, "predict_batch"):
            predictions = self.model_or_weights.predict_batch(X_with_bias)
            return np.asarray(predictions, dtype=int).reshape(-1)

        if hasattr(self.model_or_weights, "predict"):
            predictions = [self.model_or_weights.predict(sample) for sample in X_with_bias]
            return np.asarray(predictions, dtype=int).reshape(-1)

        activations = X_with_bias @ self.model_or_weights
        return np.where(activations >= 0, 1, -1).astype(int)

    def calculate_validation_metrics(self, confusion_matrix):
        return calculate_validation_metrics(confusion_matrix)


def _run_single_round(args):
    round_index, matrix, max_epochs, learning_rate = args

    train_matrix, test_matrix = create_train_test_split(matrix)
    train_matrix, test_matrix = normalize_train_test(train_matrix, test_matrix)

    initial_weights = np.random.uniform(0, 1, train_matrix.shape[1] - 1)
    perceptron = SimplePerceptron(
        train_matrix[:, :3],
        initial_weights,
        train_matrix[:, -1],
        max_epochs,
        learning_rate,
    )
    perceptron.progress_prefix = f"[Rodada {round_index}] "
    perceptron.fit()

    tester = MonteCarloTester(test_matrix, perceptron)
    confusion_matrix = tester.run_test()
    metric_values = tester.calculate_validation_metrics(confusion_matrix)

    return {
        "round": round_index,
        "confusion_matrix": confusion_matrix,
        "metrics": metric_values,
        "learning_curve": perceptron.learning_curve,
    }


def _append_round_results(results, round_result):
    results.add_record(
        round_result["round"],
        round_result["confusion_matrix"],
        round_result["metrics"],
        round_result["learning_curve"],
    )
    print(f"[Rodada {round_result['round']}] concluida")


def run_monte_carlo_validation(
    matrix,
    metric_keys,
    rounds,
    max_epochs,
    learning_rate,
    parallel,
    max_workers,
):
    results = MonteCarloResults(metric_keys)
    round_args = [
        (round_index, matrix, max_epochs, learning_rate)
        for round_index in range(1, rounds + 1)
    ]

    use_parallel = parallel and rounds > 1
    if max_workers == 1:
        use_parallel = False

    if use_parallel:
        worker_count = max_workers or os.cpu_count() or 1
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                round_results = executor.map(_run_single_round, round_args)
                for round_result in round_results:
                    _append_round_results(results, round_result)
        except (OSError, PermissionError):
            for round_args_item in round_args:
                round_result = _run_single_round(round_args_item)
                _append_round_results(results, round_result)
    else:
        for round_args_item in round_args:
            round_result = _run_single_round(round_args_item)
            _append_round_results(results, round_result)

    return results

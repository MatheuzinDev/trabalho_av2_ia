import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dados import create_train_test_split, normalize_train_test
from metricas import calculate_validation_metrics, summarize_metric_values
from modelos.adaline import Adaline
from modelos.mlp import MultilayerPerceptron
from modelos.perceptron import SimplePerceptron


def format_duration(seconds):
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes:02d}m{seconds:02d}s"
    return f"{seconds:02d}s"


class MonteCarloResults:
    def __init__(self, model_keys, metric_keys):
        self.model_keys = tuple(model_keys)
        self.metric_keys = tuple(metric_keys)
        self.metrics = {
            model_key: {metric_key: [] for metric_key in self.metric_keys}
            for model_key in self.model_keys
        }
        self.records = {model_key: [] for model_key in self.model_keys}

    def add_record(self, model_key, round_index, confusion_matrix, metric_values, learning_curve):
        record = {
            "round": round_index,
            "confusion_matrix": np.asarray(confusion_matrix, dtype=int),
            "metrics": metric_values,
            "learning_curve": list(learning_curve),
        }
        self.records[model_key].append(record)

        for metric_key in self.metric_keys:
            self.metrics[model_key][metric_key].append(metric_values[metric_key])

    def summary(self, model_key, metric_key):
        return summarize_metric_values(self.metrics[model_key][metric_key])

    def best_worst_cases(self, model_key, metric_key):
        values = np.asarray(self.metrics[model_key][metric_key], dtype=float)
        best_index = int(np.nanargmax(values))
        worst_index = int(np.nanargmin(values))
        return {
            "best": self.records[model_key][best_index],
            "worst": self.records[model_key][worst_index],
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
    (
        round_index,
        matrix,
        perceptron_max_epochs,
        perceptron_learning_rate,
        adaline_max_epochs,
        adaline_learning_rate,
        adaline_precision,
        include_mlp,
        mlp_topology,
        mlp_max_epochs,
        mlp_learning_rate,
        mlp_precision,
    ) = args

    train_matrix, test_matrix = create_train_test_split(matrix)
    train_matrix, test_matrix = normalize_train_test(train_matrix, test_matrix)

    perceptron_weights = np.random.uniform(0, 1, train_matrix.shape[1] - 1)
    perceptron = SimplePerceptron(
        train_matrix[:, :3],
        perceptron_weights,
        train_matrix[:, -1],
        perceptron_max_epochs,
        perceptron_learning_rate,
    )
    perceptron.progress_prefix = f"[Rodada {round_index}] "
    perceptron.fit()

    adaline = Adaline(train_matrix[:, 1:])
    adaline.progress_prefix = f"[Rodada {round_index}] "
    adaline.fit(adaline_max_epochs, adaline_learning_rate, adaline_precision)

    round_result = {
        "perceptron": _evaluate_model(round_index, test_matrix, perceptron),
        "adaline": _evaluate_model(round_index, test_matrix, adaline),
    }

    if include_mlp:
        X_train_mlp = train_matrix[:, 1:3].T
        Y_train_mlp = train_matrix[:, -1].reshape(1, -1)
        mlp = MultilayerPerceptron(
            mlp_topology,
            X_train_mlp,
            Y_train_mlp,
            mlp_learning_rate,
            mlp_max_epochs,
            mlp_precision,
        )
        mlp.fit()
        round_result["mlp"] = _evaluate_model(round_index, test_matrix, mlp)

    return round_result


def _evaluate_model(round_index, test_matrix, model):
    tester = MonteCarloTester(test_matrix, model)
    confusion_matrix = tester.run_test()
    metric_values = tester.calculate_validation_metrics(confusion_matrix)

    return {
        "round": round_index,
        "confusion_matrix": confusion_matrix,
        "metrics": metric_values,
        "learning_curve": getattr(model, "learning_curve", []),
    }


def _append_round_results(results, round_result, completed_rounds, total_rounds, start_time):
    for model_key, model_result in round_result.items():
        results.add_record(
            model_key,
            model_result["round"],
            model_result["confusion_matrix"],
            model_result["metrics"],
            model_result["learning_curve"],
        )

    elapsed_seconds = time.perf_counter() - start_time
    average_seconds_per_round = elapsed_seconds / completed_rounds
    remaining_rounds = total_rounds - completed_rounds
    estimated_remaining_seconds = average_seconds_per_round * remaining_rounds

    print(
        f"[Rodada {next(iter(round_result.values()))['round']}/{total_rounds}] concluida | "
        f"decorrido: {format_duration(elapsed_seconds)} | "
        f"restante estimado: {format_duration(estimated_remaining_seconds)}"
    )


def run_monte_carlo_validation(
    matrix,
    model_keys,
    metric_keys,
    rounds,
    perceptron_max_epochs,
    perceptron_learning_rate,
    adaline_max_epochs,
    adaline_learning_rate,
    adaline_precision,
    include_mlp,
    mlp_topology,
    mlp_max_epochs,
    mlp_learning_rate,
    mlp_precision,
    parallel,
    max_workers,
):
    results = MonteCarloResults(model_keys, metric_keys)
    start_time = time.perf_counter()
    completed_rounds = 0
    round_args = [
        (
            round_index,
            matrix,
            perceptron_max_epochs,
            perceptron_learning_rate,
            adaline_max_epochs,
            adaline_learning_rate,
            adaline_precision,
            include_mlp,
            mlp_topology,
            mlp_max_epochs,
            mlp_learning_rate,
            mlp_precision,
        )
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
                    completed_rounds += 1
                    _append_round_results(results, round_result, completed_rounds, rounds, start_time)
        except (OSError, PermissionError):
            for round_args_item in round_args:
                round_result = _run_single_round(round_args_item)
                completed_rounds += 1
                _append_round_results(results, round_result, completed_rounds, rounds, start_time)
    else:
        for round_args_item in round_args:
            round_result = _run_single_round(round_args_item)
            completed_rounds += 1
            _append_round_results(results, round_result, completed_rounds, rounds, start_time)

    print(f"Monte Carlo finalizado em {format_duration(time.perf_counter() - start_time)}")

    return results

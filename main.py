from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dados import create_train_test_split, load_spiral_matrix, normalize_train_test
from modelos.adaline import Adaline
from modelos.perceptron import SimplePerceptron
from monte_carlo import run_monte_carlo_validation


def load_spiral_data(file_path):
    return load_spiral_matrix(file_path)


def create_normalized_train_test_split(matrix):
    train_matrix, test_matrix = create_train_test_split(matrix)
    return normalize_train_test(train_matrix, test_matrix)


def plot_initial_scatter(matrix, ax):
    negative_class = matrix[:, -1] == -1
    positive_class = matrix[:, -1] == 1

    ax.scatter(
        matrix[negative_class, 0],
        matrix[negative_class, 1],
        c="tab:blue",
        edgecolor="k",
        label="Classe -1",
        alpha=0.8,
    )
    ax.scatter(
        matrix[positive_class, 0],
        matrix[positive_class, 1],
        c="tab:orange",
        edgecolor="k",
        label="Classe +1",
        alpha=0.8,
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()


def plot_linear_decision_boundary(ax, W, x1_values, label, color):
    if W[2] == 0:
        return

    x2_values = (W[0] - W[1] * x1_values) / W[2]
    ax.plot(x1_values, x2_values, color=color, label=label)
    ax.legend()


def train_perceptron(train_matrix, max_epochs, learning_rate):
    initial_W = np.random.uniform(0, 1, train_matrix.shape[1] - 1)
    model = SimplePerceptron(
        train_matrix[:, :3],
        initial_W,
        train_matrix[:, -1],
        max_epochs,
        learning_rate,
    )
    model.fit()
    return model


def train_adaline(train_matrix, max_epochs, learning_rate, precision):
    model = Adaline(train_matrix[:, 1:])
    model.fit(max_epochs, learning_rate, precision)
    return model


def save_boundary_figure(normalized_matrix, weights, title, boundary_label, boundary_color, output_path):
    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot()
    plot_initial_scatter(normalized_matrix, ax)
    ax.set_title(title)

    x1_min = normalized_matrix[:, 0].min()
    x1_max = normalized_matrix[:, 0].max()
    x1_values = np.linspace(x1_min, x1_max, 100)

    plot_linear_decision_boundary(
        ax,
        weights,
        x1_values,
        boundary_label,
        boundary_color,
    )

    save_figure(figure, output_path)


def save_training_example(
    matrix,
    perceptron_boundary_dir,
    adaline_boundary_dir,
    perceptron_max_epochs,
    perceptron_learning_rate,
    adaline_max_epochs,
    adaline_learning_rate,
    adaline_precision,
):
    train_matrix, test_matrix = create_normalized_train_test_split(matrix)
    normalized_matrix = np.vstack((train_matrix, test_matrix))[:, 1:]

    perceptron = train_perceptron(train_matrix, perceptron_max_epochs, perceptron_learning_rate)
    adaline = train_adaline(train_matrix, adaline_max_epochs, adaline_learning_rate, adaline_precision)

    save_boundary_figure(
        normalized_matrix,
        perceptron.W,
        "Separacao linear encontrada pelo Perceptron",
        "Fronteira Perceptron",
        "tab:red",
        perceptron_boundary_dir / "fronteira_linear.png",
    )

    save_boundary_figure(
        normalized_matrix,
        adaline.W,
        "Separacao linear encontrada pelo ADALINE",
        "Fronteira ADALINE",
        "tab:green",
        adaline_boundary_dir / "fronteira_linear.png",
    )


def print_validation_results(results, model_specs, metric_specs):
    print("\nAnalise estatistica comparativa")

    for metric_key, metric_label, _ in metric_specs:
        print(f"\nResumo estatistico de {metric_label}")
        print(f"{'Modelo':<28} {'Media':>10} {'Desvio':>10} {'Maior':>10} {'Menor':>10}")

        for model_key, model_label, _, _ in model_specs:
            summary = results.summary(model_key, metric_key)
            print(
                f"{model_label:<28} "
                f"{summary['mean']:>10.4f} "
                f"{summary['std']:>10.4f} "
                f"{summary['max']:>10.4f} "
                f"{summary['min']:>10.4f}"
            )


def write_summary_tables(results, output_dir, model_specs, metric_specs):
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, _, metric_file_label in metric_specs:
        table_path = output_dir / f"resumo_{metric_file_label}.csv"

        with table_path.open("w", encoding="utf-8") as file:
            file.write("Modelo,Media,Desvio-Padrao,Maior Valor,Menor Valor\n")
            for model_key, model_label, _, _ in model_specs:
                summary = results.summary(model_key, metric_key)
                file.write(
                    f"{model_label},"
                    f"{summary['mean']:.6f},"
                    f"{summary['std']:.6f},"
                    f"{summary['max']:.6f},"
                    f"{summary['min']:.6f}\n"
                )


def save_best_worst_artifacts(results, output_dir, model_specs, metric_specs, case_labels):
    artifacts_dir = output_dir / "casos_extremos"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for model_key, model_label, model_folder, learning_curve_label in model_specs:
        for metric_key, metric_label, metric_file_label in metric_specs:
            cases = results.best_worst_cases(model_key, metric_key)

            for case_name, record in cases.items():
                case_label = case_labels[case_name]
                case_dir = artifacts_dir / model_folder / metric_file_label / case_label
                metric_value = record["metrics"][metric_key]
                title = (
                    f"{model_label} | {case_label} caso em {metric_label} | "
                    f"rodada {record['round']} | valor={metric_value:.4f}"
                )

                save_confusion_matrix(
                    record["confusion_matrix"],
                    title,
                    case_dir / "matriz_confusao.png",
                )
                save_learning_curve(
                    record["learning_curve"],
                    title,
                    case_dir / "curva_aprendizado.png",
                    learning_curve_label,
                )


def save_confusion_matrix(confusion_matrix, title, path):
    figure, ax = plt.subplots(figsize=(5, 4))
    matrix = np.asarray(confusion_matrix, dtype=int)
    image = ax.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred +1", "Pred -1"])
    ax.set_yticklabels(["Real +1", "Real -1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    save_figure(figure, path)


def save_learning_curve(learning_curve, title, path, y_label):
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(learning_curve)), learning_curve)
    ax.set_title(title)
    ax.set_xlabel("Epoca")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    save_figure(figure, path)


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    project_dir = Path(__file__).resolve().parent
    data_file = project_dir / "spiral_d.csv"
    boundary_output_dir = project_dir / "resultados/fronteiras"
    perceptron_boundary_dir = boundary_output_dir / "perceptron"
    adaline_boundary_dir = boundary_output_dir / "adaline"
    comparison_output_dir = project_dir / "resultados/monte_carlo/comparacao_perceptron_adaline"

    perceptron_max_epochs = 1000
    perceptron_learning_rate = 1e-1
    adaline_max_epochs = 10000
    adaline_learning_rate = 1e-2
    adaline_precision = 1e-8
    monte_carlo_rounds = 500
    run_parallel = True
    max_workers = None

    run_training_example = True
    run_monte_carlo = True
    run_summary_tables = True
    run_best_worst_artifacts = True

    case_labels = {"best": "melhor", "worst": "pior"}
    model_specs = (
        ("perceptron", "Perceptron Simples", "perceptron", "Erros"),
        ("adaline", "ADALINE", "adaline", "EQM"),
    )
    metric_specs = (
        ("accuracy", "Acuracia", "acuracia"),
        ("sensitivity", "Sensibilidade", "sensibilidade"),
        ("specificity", "Especificidade", "especificidade"),
        ("precision", "Precisao", "precisao"),
        ("f1_score", "F1-score", "f1_score"),
    )
    model_keys = tuple(model_key for model_key, _, _, _ in model_specs)
    metric_keys = tuple(metric_key for metric_key, _, _ in metric_specs)

    perceptron_boundary_dir.mkdir(parents=True, exist_ok=True)
    adaline_boundary_dir.mkdir(parents=True, exist_ok=True)
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_spiral_data(data_file)

    if run_training_example:
        save_training_example(
            matrix,
            perceptron_boundary_dir,
            adaline_boundary_dir,
            perceptron_max_epochs,
            perceptron_learning_rate,
            adaline_max_epochs,
            adaline_learning_rate,
            adaline_precision,
        )

    if run_monte_carlo:
        results = run_monte_carlo_validation(
            matrix,
            model_keys,
            metric_keys,
            monte_carlo_rounds,
            perceptron_max_epochs,
            perceptron_learning_rate,
            adaline_max_epochs,
            adaline_learning_rate,
            adaline_precision,
            run_parallel,
            max_workers,
        )
        print_validation_results(results, model_specs, metric_specs)

        if run_summary_tables:
            write_summary_tables(results, comparison_output_dir, model_specs, metric_specs)

        if run_best_worst_artifacts:
            save_best_worst_artifacts(results, comparison_output_dir, model_specs, metric_specs, case_labels)

    print(f"\nFronteiras dos modelos gravadas em: {boundary_output_dir.resolve()}")
    print(f"Comparacao Monte Carlo gravada em: {comparison_output_dir.resolve()}")


if __name__ == "__main__":
    main()

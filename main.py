from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dados import create_train_test_split, load_spiral_matrix, normalize_train_test
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


def save_training_example(matrix, output_dir, max_epochs, learning_rate):
    train_matrix, test_matrix = create_normalized_train_test_split(matrix)
    normalized_matrix = np.vstack((train_matrix, test_matrix))[:, 1:]

    perceptron = train_perceptron(train_matrix, max_epochs, learning_rate)

    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot()
    plot_initial_scatter(normalized_matrix, ax)
    ax.set_title("Dados normalizados e fronteira linear")

    x1_min = normalized_matrix[:, 0].min()
    x1_max = normalized_matrix[:, 0].max()
    x1_values = np.linspace(x1_min, x1_max, 100)

    plot_linear_decision_boundary(
        ax,
        perceptron.W,
        x1_values,
        "Fronteira Perceptron",
        "tab:red",
    )

    save_figure(figure, output_dir / "01_dados_normalizados_fronteira.png")


def print_validation_results(results, model_label, metric_specs):
    for metric_key, metric_label, _ in metric_specs:
        print(f"\n\n------- {metric_label.upper()} -------")
        print(f"{'Modelo':<28} {'Media':>10} {'Desvio':>10} {'Maior':>10} {'Menor':>10}")

        summary = results.summary(metric_key)
        print(
            f"{model_label:<28} "
            f"{summary['mean']:>10.4f} "
            f"{summary['std']:>10.4f} "
            f"{summary['max']:>10.4f} "
            f"{summary['min']:>10.4f}"
        )


def write_summary_tables(results, output_dir, model_label, metric_specs):
    for metric_key, _, metric_file_label in metric_specs:
        table_path = output_dir / f"tabela_{metric_file_label}.csv"

        with table_path.open("w", encoding="utf-8") as file:
            file.write("Modelo,Media,Desvio-Padrao,Maior Valor,Menor Valor\n")
            summary = results.summary(metric_key)
            file.write(
                f"{model_label},"
                f"{summary['mean']:.6f},"
                f"{summary['std']:.6f},"
                f"{summary['max']:.6f},"
                f"{summary['min']:.6f}\n"
            )


def save_metric_boxplots(results, output_dir, model_label, metric_specs):
    for metric_key, metric_label, metric_file_label in metric_specs:

        figure, ax = plt.subplots(figsize=(8, 5))
        values = [results.metrics[metric_key]]
        labels = [model_label]

        ax.boxplot(values, labels=labels)
        ax.set_title(f"Distribuicao - {metric_label}")
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.3)
        figure.autofmt_xdate(rotation=15)

        save_figure(figure, output_dir / f"grafico_caixa_{metric_file_label}.png")


def save_best_worst_artifacts(results, output_dir, model_label, metric_specs, case_labels):
    artifacts_dir = output_dir / "melhores_piores"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_label, metric_file_label in metric_specs:
        cases = results.best_worst_cases(metric_key)

        for case_name, record in cases.items():
            case_label = case_labels[case_name]
            prefix = f"perceptron_{metric_file_label}_{case_label}_rodada_{record['round']}"
            metric_value = record["metrics"][metric_key]
            title = f"{model_label} - {case_label} {metric_label} = {metric_value:.4f}"

            save_confusion_matrix(
                record["confusion_matrix"],
                title,
                artifacts_dir / f"{prefix}_matriz_confusao.png",
            )
            save_learning_curve(
                record["learning_curve"],
                title,
                artifacts_dir / f"{prefix}_curva_aprendizado.png",
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


def save_learning_curve(learning_curve, title, path):
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(learning_curve)), learning_curve)
    ax.set_title(title)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Erros")
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
    output_dir = project_dir / "resultados/perceptron"
    model_label = "Perceptron Simples"

    max_epochs = 10000
    learning_rate = 1e-2
    monte_carlo_rounds = 500
    run_parallel = True
    max_workers = None

    run_training_example = True
    run_monte_carlo = False
    run_summary_tables = True
    run_metric_boxplots = True
    run_best_worst_artifacts = True

    case_labels = {"best": "melhor", "worst": "pior"}
    metric_specs = (
        ("accuracy", "Acuracia", "acuracia"),
        ("sensitivity", "Sensibilidade", "sensibilidade"),
        ("specificity", "Especificidade", "especificidade"),
        ("precision", "Precisao", "precisao"),
        ("f1_score", "F1-score", "f1_score"),
    )
    metric_keys = tuple(metric_key for metric_key, _, _ in metric_specs)

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_spiral_data(data_file)

    if run_training_example:
        save_training_example(matrix, output_dir, max_epochs, learning_rate)

    if run_monte_carlo:
        results = run_monte_carlo_validation(
            matrix,
            metric_keys,
            monte_carlo_rounds,
            max_epochs,
            learning_rate,
            run_parallel,
            max_workers,
        )
        print_validation_results(results, model_label, metric_specs)

        if run_summary_tables:
            write_summary_tables(results, output_dir, model_label, metric_specs)

        if run_metric_boxplots:
            save_metric_boxplots(results, output_dir, model_label, metric_specs)

        if run_best_worst_artifacts:
            save_best_worst_artifacts(results, output_dir, model_label, metric_specs, case_labels)

    print(f"\nArtefatos salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

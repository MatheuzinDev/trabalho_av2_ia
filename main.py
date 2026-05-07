from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dados import create_train_test_split, load_spiral_matrix, normalize_train_test
from metricas import METRIC_FILE_LABELS, METRIC_KEYS, METRIC_LABELS
from modelos.perceptron import SimplePerceptron
from monte_carlo import run_monte_carlo_validation


DATA_FILE = "spiral_d.csv"
OUTPUT_DIR = Path("resultados/perceptron")
MODEL_LABEL = "Perceptron Simples"

PERCEPTRON_MAX_EPOCHS = 10000
PERCEPTRON_LEARNING_RATE = 1e-2

MONTE_CARLO_ROUNDS = 500

RUN_TRAINING_EXAMPLE = True
RUN_MONTE_CARLO = False
RUN_SUMMARY_TABLES = True
RUN_METRIC_BOXPLOTS = True
RUN_BEST_WORST_ARTIFACTS = True
CASE_LABELS = {"best": "melhor", "worst": "pior"}


def load_spiral_data(file_name=DATA_FILE):
    return load_spiral_matrix(file_name)


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


def train_perceptron(train_matrix):
    initial_W = np.random.uniform(0, 1, train_matrix.shape[1] - 1)
    model = SimplePerceptron(
        train_matrix[:, :3],
        initial_W,
        train_matrix[:, -1],
        PERCEPTRON_MAX_EPOCHS,
        PERCEPTRON_LEARNING_RATE,
    )
    model.fit()
    return model


def save_training_example(matrix, output_dir):
    train_matrix, test_matrix = create_normalized_train_test_split(matrix)
    normalized_matrix = np.vstack((train_matrix, test_matrix))[:, 1:]

    perceptron = train_perceptron(train_matrix)

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


def print_validation_results(results):
    for metric_key in METRIC_KEYS:
        print(f"\n\n------- {METRIC_LABELS[metric_key].upper()} -------")
        print(f"{'Modelo':<28} {'Media':>10} {'Desvio':>10} {'Maior':>10} {'Menor':>10}")

        summary = results.summary(metric_key)
        print(
            f"{MODEL_LABEL:<28} "
            f"{summary['mean']:>10.4f} "
            f"{summary['std']:>10.4f} "
            f"{summary['max']:>10.4f} "
            f"{summary['min']:>10.4f}"
        )


def write_summary_tables(results, output_dir):
    for metric_key in METRIC_KEYS:
        metric_file_label = METRIC_FILE_LABELS[metric_key]
        table_path = output_dir / f"tabela_{metric_file_label}.csv"

        with table_path.open("w", encoding="utf-8") as file:
            file.write("Modelo,Media,Desvio-Padrao,Maior Valor,Menor Valor\n")
            summary = results.summary(metric_key)
            file.write(
                f"{MODEL_LABEL},"
                f"{summary['mean']:.6f},"
                f"{summary['std']:.6f},"
                f"{summary['max']:.6f},"
                f"{summary['min']:.6f}\n"
            )


def save_metric_boxplots(results, output_dir):
    for metric_key in METRIC_KEYS:
        metric_file_label = METRIC_FILE_LABELS[metric_key]

        figure, ax = plt.subplots(figsize=(8, 5))
        values = [results.metrics[metric_key]]
        labels = [MODEL_LABEL]

        ax.boxplot(values, labels=labels)
        ax.set_title(f"Distribuicao - {METRIC_LABELS[metric_key]}")
        ax.set_ylabel(METRIC_LABELS[metric_key])
        ax.grid(axis="y", alpha=0.3)
        figure.autofmt_xdate(rotation=15)

        save_figure(figure, output_dir / f"grafico_caixa_{metric_file_label}.png")


def save_best_worst_artifacts(results, output_dir):
    artifacts_dir = output_dir / "melhores_piores"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for metric_key in METRIC_KEYS:
        metric_file_label = METRIC_FILE_LABELS[metric_key]
        cases = results.best_worst_cases(metric_key)

        for case_name, record in cases.items():
            case_label = CASE_LABELS[case_name]
            prefix = f"perceptron_{metric_file_label}_{case_label}_rodada_{record['round']}"
            metric_value = record["metrics"][metric_key]
            title = f"{MODEL_LABEL} - {case_label} {METRIC_LABELS[metric_key]} = {metric_value:.4f}"

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = load_spiral_data()

    if RUN_TRAINING_EXAMPLE:
        save_training_example(matrix, OUTPUT_DIR)

    if RUN_MONTE_CARLO:
        results = run_monte_carlo_validation(
            matrix,
            rounds=MONTE_CARLO_ROUNDS,
            max_epochs=PERCEPTRON_MAX_EPOCHS,
            learning_rate=PERCEPTRON_LEARNING_RATE,
        )
        print_validation_results(results)

        if RUN_SUMMARY_TABLES:
            write_summary_tables(results, OUTPUT_DIR)

        if RUN_METRIC_BOXPLOTS:
            save_metric_boxplots(results, OUTPUT_DIR)

        if RUN_BEST_WORST_ARTIFACTS:
            save_best_worst_artifacts(results, OUTPUT_DIR)

    print(f"\nArtefatos salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

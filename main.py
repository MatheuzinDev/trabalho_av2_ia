from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dados import create_train_test_split, load_spiral_matrix, normalize_train_test
from modelos.adaline import Adaline
from modelos.mlp import MultilayerPerceptron
from modelos.perceptron import SimplePerceptron
from monte_carlo import MonteCarloTester, run_monte_carlo_validation


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
        c="tab:red",
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


def train_mlp(train_matrix, topology, max_epochs, learning_rate, precision):
    X_train = train_matrix[:, 1:3].T
    Y_train = train_matrix[:, -1].reshape(1, -1)

    model = MultilayerPerceptron(
        topology,
        X_train,
        Y_train,
        learning_rate,
        max_epochs,
        precision,
    )
    model.fit()
    return model


def plot_mlp_decision_boundary(ax, mlp, normalized_matrix, grid_size=250):
    x_min = normalized_matrix[:, 0].min()
    x_max = normalized_matrix[:, 0].max()
    y_min = normalized_matrix[:, 1].min()
    y_max = normalized_matrix[:, 1].max()
    margin = 0.05

    xx, yy = np.meshgrid(
        np.linspace(x_min - margin, x_max + margin, grid_size),
        np.linspace(y_min - margin, y_max + margin, grid_size),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.predict_batch(grid_points).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        predictions,
        levels=[-1, 0, 1],
        colors=["tab:blue", "tab:red"],
        alpha=0.18,
    )
    ax.contour(xx, yy, predictions, levels=[0], colors="black", linewidths=1.5)


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


def save_mlp_boundary_figure(normalized_matrix, mlp, title, output_path):
    figure, ax = plt.subplots(figsize=(8, 6))
    plot_mlp_decision_boundary(ax, mlp, normalized_matrix)
    plot_initial_scatter(normalized_matrix, ax)
    ax.set_title(title)
    ax.set_xlabel("x1 normalizado")
    ax.set_ylabel("x2 normalizado")
    save_figure(figure, output_path)


def save_initial_data_scatter(matrix, output_path):
    figure, ax = plt.subplots(figsize=(8, 6))
    plot_initial_scatter(matrix, ax)
    ax.set_title("Distribuicao inicial dos dados")
    ax.grid(alpha=0.3)
    save_figure(figure, output_path)


def save_training_example(
    matrix,
    perceptron_boundary_dir,
    adaline_boundary_dir,
    mlp_boundary_dir,
    perceptron_max_epochs,
    perceptron_learning_rate,
    adaline_max_epochs,
    adaline_learning_rate,
    adaline_precision,
    run_mlp_training_example,
    mlp_topology,
    mlp_max_epochs,
    mlp_learning_rate,
    mlp_precision,
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
        "tab:purple",
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

    if run_mlp_training_example:
        mlp = train_mlp(train_matrix, mlp_topology, mlp_max_epochs, mlp_learning_rate, mlp_precision)
        save_mlp_boundary_figure(
            normalized_matrix,
            mlp,
            "Fronteira de decisao encontrada pela MLP",
            mlp_boundary_dir / "fronteira_decisao.png",
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
    figure, ax = plt.subplots(figsize=(6, 5))
    matrix = np.asarray(confusion_matrix, dtype=int)

    cells = (
        (0, 0, "Verdadeiro\nPositivo\n[1,1]", matrix[0, 0], "#d8ead3"),
        (1, 0, "Falso\nPositivo\n[1,-1]", matrix[0, 1], "#eda5ad"),
        (0, 1, "Falso\nNegativo\n[-1,1]", matrix[1, 0], "#eda5ad"),
        (1, 1, "Verdadeiro\nNegativo\n[-1,-1]", matrix[1, 1], "#d8ead3"),
    )

    for column, row, label, value, color in cells:
        ax.add_patch(Rectangle((column, row), 1, 1, facecolor=color, edgecolor="black"))
        ax.text(
            column + 0.5,
            row + 0.5,
            f"{label}\n{value}",
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.set_title(title, pad=55, fontsize=9)
    ax.set_xlabel("Real", fontsize=14, labelpad=12)
    ax.set_ylabel("Predito", fontsize=14, labelpad=18)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Condicao\nPositiva", "Condicao\nNegativa"])
    ax.set_yticklabels(["Condicao\nPositiva", "Condicao\nNegativa"], rotation=90, va="center")
    ax.tick_params(length=0)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(False)

    save_figure(figure, path)


def save_learning_curve(learning_curve, title, path, y_label):
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(learning_curve)), learning_curve)
    ax.set_title(title)
    ax.set_xlabel("Epoca")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    save_figure(figure, path)


def format_topology(topology):
    return "x".join(str(neuron_count) for neuron_count in topology)


def save_mlp_topology_study(
    matrix,
    output_dir,
    topology_study,
    max_epochs,
    learning_rate,
    precision,
    metric_specs,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_matrix, test_matrix = create_normalized_train_test_split(matrix)
    normalized_matrix = np.vstack((train_matrix, test_matrix))[:, 1:]
    table_path = output_dir / "resumo_topologias.csv"

    with table_path.open("w", encoding="utf-8") as file:
        header = ["Caso", "Topologia"] + [metric_label for _, metric_label, _ in metric_specs]
        file.write(",".join(header) + "\n")

        for case_name, topology in topology_study.items():
            mlp = train_mlp(train_matrix, topology, max_epochs, learning_rate, precision)
            tester = MonteCarloTester(test_matrix, mlp)
            confusion_matrix = tester.run_test()
            metrics = tester.calculate_validation_metrics(confusion_matrix)
            topology_label = format_topology(topology)
            case_dir = output_dir / case_name
            title = f"MLP {case_name} | topologia {topology_label}"

            values = [case_name, topology_label]
            values.extend(f"{metrics[metric_key]:.6f}" for metric_key, _, _ in metric_specs)
            file.write(",".join(values) + "\n")

            save_confusion_matrix(
                confusion_matrix,
                title,
                case_dir / "matriz_confusao.png",
            )
            save_learning_curve(
                mlp.learning_curve,
                title,
                case_dir / "curva_aprendizado.png",
                "EQM",
            )
            save_mlp_boundary_figure(
                normalized_matrix,
                mlp,
                title,
                case_dir / "fronteira_decisao.png",
            )


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
    mlp_boundary_dir = boundary_output_dir / "mlp"
    mlp_topology_output_dir = project_dir / "resultados/mlp/topologias"
    comparison_output_dir = project_dir / "resultados/monte_carlo/comparacao_modelos"

    perceptron_max_epochs = 10000
    perceptron_learning_rate = 1e-2
    adaline_max_epochs = 10000
    adaline_learning_rate = 1e-2
    adaline_precision = 1e-8
    mlp_topology = (10,)
    mlp_max_epochs = 10000
    mlp_learning_rate = 1e-2
    mlp_precision = 1e-8
    mlp_topology_study = {
        "subdimensionado": (2,),
        "referencia": mlp_topology,
        "superdimensionado": (100, 100),
    }
    monte_carlo_rounds = 500
    run_parallel = True
    max_workers = None

    run_training_example = False
    run_mlp_training_example = True
    run_mlp_topology_study = True
    run_monte_carlo = False
    include_mlp_in_monte_carlo = True
    run_summary_tables = True
    run_best_worst_artifacts = True

    case_labels = {"best": "melhor", "worst": "pior"}
    model_specs = [
        ("perceptron", "Perceptron Simples", "perceptron", "Erros"),
        ("adaline", "ADALINE", "adaline", "EQM"),
    ]
    if include_mlp_in_monte_carlo:
        model_specs.append(("mlp", "MLP", "mlp", "EQM"))
    model_specs = tuple(model_specs)
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
    mlp_boundary_dir.mkdir(parents=True, exist_ok=True)
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_spiral_data(data_file)
    save_initial_data_scatter(matrix, boundary_output_dir / "dados_iniciais.png")

    if run_training_example:
        save_training_example(
            matrix,
            perceptron_boundary_dir,
            adaline_boundary_dir,
            mlp_boundary_dir,
            perceptron_max_epochs,
            perceptron_learning_rate,
            adaline_max_epochs,
            adaline_learning_rate,
            adaline_precision,
            run_mlp_training_example,
            mlp_topology,
            mlp_max_epochs,
            mlp_learning_rate,
            mlp_precision,
        )

    if run_mlp_topology_study:
        save_mlp_topology_study(
            matrix,
            mlp_topology_output_dir,
            mlp_topology_study,
            mlp_max_epochs,
            mlp_learning_rate,
            mlp_precision,
            metric_specs,
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
            include_mlp_in_monte_carlo,
            mlp_topology,
            mlp_max_epochs,
            mlp_learning_rate,
            mlp_precision,
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

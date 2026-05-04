from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from monte_carlos import run_test
from perceptron_simples import LinearPerceptron


def load_spiral_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    X = data[:, :2]
    y = data[:, 2]

    return X, y


def plot_initial_scatter(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1.0, 0], X[y == 1.0, 1], s=18, alpha=0.75, label="Classe +1")
    plt.scatter(X[y == -1.0, 0], X[y == -1.0, 1], s=18, alpha=0.75, label="Classe -1")
    plt.title("Distribuicao inicial do conjunto spiral_d")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()

def create_perceptron(X_train, y_train):
    return LinearPerceptron(
        X_train=X_train.T,
        y_train=y_train.reshape(1, -1),
        learning_rate=0.01,
        max_epochs=200,
    )


def predict_perceptron(model, X_test):
    return model.predict(X_test.T).ravel()


def print_summary(results):
    for metric_name, values in results["resumo"].items():
        print(f"\n{metric_name.upper()}")
        print(f"Media: {values['media']:.4f}")
        print(f"Desvio-padrao: {values['desvio_padrao']:.4f}")
        print(f"Maior valor: {values['maior']:.4f}")
        print(f"Menor valor: {values['menor']:.4f}")


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_path = project_dir / "spiral_d.csv"

    X, y = load_spiral_dataset(dataset_path)
    plot_initial_scatter(X, y)

    results = run_test(
        X,
        y,
        R=500,
        create_model=create_perceptron,
        predict_model=predict_perceptron,
    )

    print("Perceptron Simples - Monte Carlo")
    print("Rodadas: 500")
    print_summary(results)

    plt.show()


if __name__ == "__main__":
    main()

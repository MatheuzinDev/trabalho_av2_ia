from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron_simples import LinearPerceptron


def load_spiral_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    X = data[:, :2]
    y = data[:, 2]
    return X, y


def train_test_split_random(X, y, test_ratio=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(X.shape[0])
    test_size = int(np.floor(X.shape[0] * test_ratio))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    return X_train_std, X_test_std, mean, std


def confusion_matrix_binary(y_true, y_pred, positive_label=1.0):
    negative_label = -1.0 if positive_label == 1.0 else 1.0
    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true == negative_label) & (y_pred == negative_label)))
    fp = int(np.sum((y_true == negative_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred == negative_label)))
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


def plot_initial_scatter(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1.0, 0], X[y == 1.0, 1], s=18, alpha=0.75, label="Classe +1")
    plt.scatter(X[y == -1.0, 0], X[y == -1.0, 1], s=18, alpha=0.75, label="Classe -1")
    plt.title("Distribuicao inicial do conjunto spiral_d")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()


def plot_learning_curve(model):
    epochs = np.arange(1, len(model.errors_per_epoch_) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, model.errors_per_epoch_, marker="o")
    axes[0].set_title("Erros por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Quantidade de erros")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, model.accuracy_per_epoch_, marker="o", color="tab:green")
    axes[1].set_title("Acuracia no treino por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()


def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1.0, 0], X[y == 1.0, 1], s=18, alpha=0.75, label="Classe +1")
    plt.scatter(X[y == -1.0, 0], X[y == -1.0, 1], s=18, alpha=0.75, label="Classe -1")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_values = np.linspace(x_min, x_max, 200)

    bias_weight = model.w[0, 0]
    x1_weight = model.w[1, 0]
    x2_weight = model.w[2, 0]

    if abs(x2_weight) > 1e-12:
        y_values = (bias_weight - x1_weight * x_values) / x2_weight
        plt.plot(x_values, y_values, color="black", linewidth=2, label="Fronteira do Perceptron")

    plt.title("Fronteira de decisao do Perceptron no treino")
    plt.xlabel("x1 normalizado")
    plt.ylabel("x2 normalizado")
    plt.legend()
    plt.tight_layout()


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_path = project_dir / "spiral_d.csv"

    X, y = load_spiral_dataset(dataset_path)
    plot_initial_scatter(X, y)

    X_train, X_test, y_train, y_test = train_test_split_random(
        X,
        y,
        test_ratio=0.2,
        random_state=42,
    )
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)

    model = LinearPerceptron(
        X_train=X_train_std.T,
        y_train=y_train.reshape(1, -1),
        learning_rate=0.01,
        max_epochs=200,
    )
    model.fit()

    y_pred = model.predict(X_test_std.T).ravel()
    confusion_matrix = confusion_matrix_binary(y_test, y_pred)
    metrics = compute_metrics(confusion_matrix)

    plot_learning_curve(model)
    plot_decision_boundary(model, X_train_std, y_train)

    print("Perceptron Simples - Primeira execucao")
    print(f"Amostras de treino: {X_train.shape[0]}")
    print(f"Amostras de teste: {X_test.shape[0]}")
    print(f"Epocas executadas: {len(model.errors_per_epoch_)}")
    print(f"Convergiu: {'sim' if model.converged_ else 'nao'}")
    print("Matriz de confusao [[TN, FP], [FN, TP]]:")
    print(confusion_matrix)
    print(f"Acuracia: {metrics['acuracia']:.4f}")
    print(f"Sensibilidade: {metrics['sensibilidade']:.4f}")
    print(f"Especificidade: {metrics['especificidade']:.4f}")
    print(f"Precisao: {metrics['precisao']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")

    plt.show()


if __name__ == "__main__":
    main()

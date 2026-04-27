import numpy as np


class LinearPerceptron:
    def __init__(self, X_train, y_train, learning_rate, max_epochs=200):
        self.X_features = np.asarray(X_train, dtype=float)
        self.p, self.N = self.X_features.shape
        self.d = np.asarray(y_train, dtype=float)
        self.lr = learning_rate
        self.max_epochs = max_epochs

        self.X_train = np.vstack((
            -np.ones((1, self.N)),
            self.X_features,
        ))

        self.w = np.random.random_sample((self.p + 1, 1)) - 0.5
        self.errors_per_epoch_ = []
        self.accuracy_per_epoch_ = []
        self.converged_ = False

    @staticmethod
    def activation_function(u):
        return 1 if u >= 0 else -1

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X deve ter formato (p, N).")
        if X.shape[0] != self.p:
            raise ValueError("X deve ter a mesma quantidade de atributos usada no treino.")

        X_with_bias = np.vstack((-np.ones((1, X.shape[1])), X))
        return (self.w.T @ X_with_bias).ravel()

    def predict(self, X):
        activations = self.decision_function(X)
        predictions = np.where(activations >= 0.0, 1.0, -1.0)
        return predictions.reshape(1, -1)

    def fit(self):
        epoch = 0
        error = True

        while error and epoch < self.max_epochs:
            error = False
            errors = 0

            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                u_k = float((self.w.T @ x_k)[0, 0])
                d_k = float(self.d[0, k])
                y_k = self.activation_function(u_k)
                e_k = d_k - y_k

                if e_k != 0.0:
                    error = True
                    self.w = self.w + self.lr * e_k * x_k
                    errors += 1

            train_predictions = self.predict(self.X_features)
            train_accuracy = np.mean(train_predictions == self.d)

            self.errors_per_epoch_.append(errors)
            self.accuracy_per_epoch_.append(float(train_accuracy))

            if not error:
                self.converged_ = True

            epoch += 1

        return self

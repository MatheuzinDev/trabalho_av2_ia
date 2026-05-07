import numpy as np


class SimplePerceptron:
    def __init__(self, input_matrix, initial_weights, target_values, max_epochs=10000, learning_rate=0.01):
        self.X_train = input_matrix
        self.W = initial_weights
        self.y = target_values
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.learning_curve = []

    def _bipolar_step_activation(self, x):
        if x >= 0:
            return 1
        return -1

    def fit(self):
        epochs = 0
        learning_rate = self.learning_rate
        has_error = True

        while has_error and epochs < self.max_epochs:
            has_error = False
            epoch_errors = 0

            for sample_index in range(self.X_train.shape[0]):
                x_k = self.X_train[sample_index]
                u_k = np.dot(self.W, x_k)
                y_k = self._bipolar_step_activation(u_k)
                d_k = self.y[sample_index]

                if y_k != d_k:
                    self.W = self.W + (learning_rate * (d_k - y_k) * x_k)
                    has_error = True
                    epoch_errors += 1

            epochs += 1
            self.learning_curve.append(epoch_errors)

        return self

    def predict(self, x_with_bias):
        u = np.dot(self.W, x_with_bias)
        return self._bipolar_step_activation(u)

    def predict_batch(self, X_with_bias):
        activations = np.asarray(X_with_bias) @ self.W
        return np.where(activations >= 0, 1, -1)

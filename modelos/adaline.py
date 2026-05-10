import numpy as np


class Adaline:
    def __init__(self, training_matrix):
        self.training_matrix = training_matrix
        self.W = None
        self.learning_curve = []

    def _bipolar_step_activation(self, value):
        if value >= 0:
            return 1
        return -1

    def _calculate_mean_squared_error(self, sample_count, weights, input_matrix):
        predictions = input_matrix @ weights
        targets = self.training_matrix[:, -1]
        errors = targets - predictions
        return np.sum(errors**2) / sample_count

    def fit(self, max_epochs, learning_rate, precision):
        input_matrix = self.training_matrix[:, :2]
        bias_column = -np.ones(input_matrix.shape[0]).reshape(-1, 1)
        input_matrix = np.hstack((bias_column, input_matrix))

        weights = np.random.uniform(0, 1, self.training_matrix.shape[1])

        epochs = 0
        precision_reached = False
        current_mse = self._calculate_mean_squared_error(
            self.training_matrix.shape[0],
            weights,
            input_matrix,
        )
        self.learning_curve = [current_mse]

        while epochs < max_epochs and not precision_reached:
            previous_mse = current_mse

            for sample_index in range(self.training_matrix.shape[0]):
                x_k = input_matrix[sample_index]
                linear_output = np.dot(weights, x_k)
                target = self.training_matrix[sample_index, -1]

                weights = weights + learning_rate * (target - linear_output) * x_k

            epochs += 1
            current_mse = self._calculate_mean_squared_error(
                self.training_matrix.shape[0],
                weights,
                input_matrix,
            )
            self.learning_curve.append(current_mse)

            if abs(current_mse - previous_mse) <= precision:
                precision_reached = True

        self.W = weights
        return self

    def predict(self, x_with_bias):
        linear_output = np.dot(self.W, x_with_bias)
        return self._bipolar_step_activation(linear_output)

    def predict_batch(self, X_with_bias):
        activations = np.asarray(X_with_bias) @ self.W
        return np.where(activations >= 0, 1, -1)

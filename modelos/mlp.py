import numpy as np


class MultilayerPerceptron:
    def __init__(
        self,
        topology,
        X_train,
        Y_train,
        learning_rate,
        max_epochs,
        precision,
    ):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.precision = precision
        self.learning_curve = []

        self.feature_count, self.sample_count = X_train.shape
        self.output_dim = Y_train.shape[0]
        self.topology = list(topology) + [self.output_dim]

        self.X_train = np.vstack((-np.ones((1, self.sample_count)), X_train))
        self.D = Y_train

        self.W = []
        input_dim = self.feature_count + 1
        for neuron_count in self.topology:
            weights = np.random.random_sample((neuron_count, input_dim)) - 0.5
            self.W.append(weights)
            input_dim = neuron_count + 1

        self.u = [None] * len(self.W)
        self.y = [None] * len(self.W)
        self.delta = [None] * len(self.W)

    def _activation(self, u):
        return np.tanh(u / 2.0)

    def _activation_derivative(self, u):
        activated = self._activation(u)
        return 0.5 * (1 - activated**2)

    def _forward_batch(self, X_with_bias):
        activations = X_with_bias
        for layer_index, weights in enumerate(self.W):
            u = weights @ activations
            y = self._activation(u)
            if layer_index < len(self.W) - 1:
                activations = np.vstack((-np.ones((1, y.shape[1])), y))
            else:
                activations = y
        return activations

    def mean_squared_error(self):
        predictions = self._forward_batch(self.X_train)
        return np.sum((self.D - predictions) ** 2) / (2 * self.sample_count)

    def forward(self, x):
        for layer_index, weights in enumerate(self.W):
            if layer_index == 0:
                self.u[layer_index] = weights @ x
            else:
                previous_output = np.vstack((-np.ones((1, 1)), self.y[layer_index - 1]))
                self.u[layer_index] = weights @ previous_output

            self.y[layer_index] = self._activation(self.u[layer_index])

        return self.y[-1]

    def backward(self, x, target):
        for layer_index in range(len(self.W) - 1, -1, -1):
            if layer_index == len(self.W) - 1:
                self.delta[layer_index] = self._activation_derivative(self.u[layer_index]) * (
                    target - self.y[layer_index]
                )
            else:
                next_weights_without_bias = self.W[layer_index + 1][:, 1:].T
                self.delta[layer_index] = self._activation_derivative(self.u[layer_index]) * (
                    next_weights_without_bias @ self.delta[layer_index + 1]
                )

            if layer_index == 0:
                input_with_bias = x
            else:
                input_with_bias = np.vstack((-np.ones((1, 1)), self.y[layer_index - 1]))

            self.W[layer_index] = self.W[layer_index] + (
                self.learning_rate * self.delta[layer_index] @ input_with_bias.T
            )

    def fit(self):
        epochs = 0
        current_mse = self.mean_squared_error()
        self.learning_curve = [current_mse]

        while epochs < self.max_epochs and current_mse > self.precision:
            for sample_index in range(self.sample_count):
                x_k = self.X_train[:, sample_index].reshape(self.feature_count + 1, 1)
                d_k = self.D[:, sample_index].reshape(self.output_dim, 1)
                self.forward(x_k)
                self.backward(x_k, d_k)

            epochs += 1
            current_mse = self.mean_squared_error()
            self.learning_curve.append(current_mse)

        return self

    def _prepare_input_matrix(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] == self.feature_count + 1:
            X = X[:, 1:]
        elif X.shape[1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} or {self.feature_count + 1} features, received {X.shape[1]}")

        return X

    def predict_raw(self, x):
        X = self._prepare_input_matrix(x)
        x_column = X.reshape(self.feature_count, 1)
        x_with_bias = np.vstack((-np.ones((1, 1)), x_column))
        return self.forward(x_with_bias)

    def predict_raw_batch(self, X):
        X = self._prepare_input_matrix(X)
        X_transposed = X.T
        X_with_bias = np.vstack((-np.ones((1, X_transposed.shape[1])), X_transposed))
        return self._forward_batch(X_with_bias)

    def predict(self, x):
        raw_output = self.predict_raw(x)
        return np.where(raw_output >= 0, 1, -1).reshape(-1)

    def predict_batch(self, X):
        raw_output = self.predict_raw_batch(X)
        return np.where(raw_output >= 0, 1, -1).reshape(-1)

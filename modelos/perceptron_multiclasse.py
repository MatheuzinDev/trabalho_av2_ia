import numpy as np

class PerceptronMulticlasse:
    def __init__(self, num_atributos, num_classes=20, max_epochs=1000, learning_rate=0.01):
        self.W = np.random.uniform(-0.1, 0.1, (num_classes, num_atributos))
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.learning_curve = []
        self.num_classes = num_classes

    def fit(self, X_train, Y_train):
        epochs = 0
        has_error = True

        while has_error and epochs < self.max_epochs:
            has_error = False
            erros_epoca = 0

            for i in range(X_train.shape[1]):
                x_k = X_train[:, i].reshape(-1, 1)
                d_k = Y_train[:, i].reshape(-1, 1)
                
                u_k = np.dot(self.W, x_k)
                y_k = np.where(u_k >= 0, 1, -1)

                if not np.array_equal(y_k, d_k):
                    self.W = self.W + self.learning_rate * np.dot((d_k - y_k), x_k.T)
                    has_error = True
                    erros_epoca += 1

            self.learning_curve.append(erros_epoca)
            epochs += 1

        return self

    def predict(self, X_test):
        u = np.dot(self.W, X_test)
        return np.argmax(u, axis=0)
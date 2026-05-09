import numpy as np

class AdalineMulticlasse:
    def __init__(self, num_atributos, num_classes=20, max_epochs=1000, learning_rate=0.001, precision=1e-5):
        self.W = np.random.uniform(-0.01, 0.01, (num_classes, num_atributos))
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.precision = precision
        self.learning_curve = []
        self.num_classes = num_classes

    def fit(self, X_train, Y_train):
        N = X_train.shape[1]
        epochs = 0
        
        v_all = np.dot(self.W, X_train)
        current_mse = np.sum((Y_train - v_all)**2) / N
        self.learning_curve.append(current_mse)
        
        while epochs < self.max_epochs:
            previous_mse = current_mse
            
            for i in range(N):
                x_k = X_train[:, i].reshape(-1, 1)
                d_k = Y_train[:, i].reshape(-1, 1)
                
                v_k = np.dot(self.W, x_k)
                self.W = self.W + self.learning_rate * np.dot((d_k - v_k), x_k.T)
                
            v_all = np.dot(self.W, X_train)
            current_mse = np.sum((Y_train - v_all)**2) / N
            self.learning_curve.append(current_mse)
            
            if abs(previous_mse - current_mse) < self.precision:
                break
                
            epochs += 1
            
        return self

    def predict(self, X_test):
        u = np.dot(self.W, X_test)
        return np.argmax(u, axis=0)
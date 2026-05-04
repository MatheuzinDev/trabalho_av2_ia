import numpy as np

class LinearPerceptron:
    def __init__(self, X_train, y_train, learning_rate=0.5, max_epochs=100):
        self.X = X_train
        self.y = y_train
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.p, self.N = self.X.shape
        
        # adiciona o bias como primeira linha
        self.X = np.vstack((
            -np.ones((1, self.N)),
            self.X,
        ))
        
        # um peso para o bias e um para cada atributo
        self.w = np.random.random_sample((self.p + 1, 1)) - 0.5
        self.converged_ = False
        
    def activation_function(self, u):
        if u >= 0:
            return 1
        else:
            return -1
        
    def predict(self, X):
        X = np.vstack((
            -np.ones((1, X.shape[1])),
            X,
        ))
        activations = self.w.T @ X
        predictions = np.where(activations >= 0, 1, -1)
        return predictions
    
    def fit(self):
        epoch = 0
        error = True
        
        while error and epoch < self.max_epochs:
            error = False
            
            for k in range(self.N):
                x_k = self.X[:, k].reshape(self.p + 1, 1)
                u_k = self.w.T @ x_k
                y_k = self.activation_function(u_k[0, 0])
                d_k = self.y[0, k]
                
                if y_k != d_k:
                    self.w = self.w + self.learning_rate * (d_k - y_k) * x_k
                    error = True
                    
            if not error:
                self.converged_ = True
            epoch += 1
            
        return self
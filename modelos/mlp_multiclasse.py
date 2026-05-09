import numpy as np

class MLPMulticlasse:
    def __init__(self, num_entradas, num_ocultos, num_saidas=20, max_epochs=100, learning_rate=0.01, precision=1e-5):
        self.eta = learning_rate
        self.max_epochs = max_epochs
        self.precision = precision
        self.learning_curve = []
        
        self.W1 = np.random.uniform(-0.1, 0.1, (num_ocultos, num_entradas))
        
        self.W2 = np.random.uniform(-0.1, 0.1, (num_saidas, num_ocultos + 1))

    def ativacao(self, v):
        return np.tanh(v)

    def derivada_ativacao(self, y):
        return 1.0 - y**2

    def fit(self, X_train, Y_train):
        N = X_train.shape[1]
        epochs = 0
        erro_anterior = float('inf')
        
        while epochs < self.max_epochs:
            erro_epoca = 0
            
            for i in range(N):
                x_k = X_train[:, i].reshape(-1, 1) 
                d_k = Y_train[:, i].reshape(-1, 1) 
                
                # Fordward
                v1 = np.dot(self.W1, x_k)
                y1 = self.ativacao(v1)
                
                y1_bias = np.vstack(([1.0], y1)) 
                
                v2 = np.dot(self.W2, y1_bias)
                y2 = self.ativacao(v2)
                
                # Calculo do erro
                erro = d_k - y2
                erro_epoca += np.sum(erro**2)
                
                # Backward
                delta2 = erro * self.derivada_ativacao(y2)
                # Ignora a linha do bias de W2 na hora de propagar o erro
                delta1 = np.dot(self.W2[:, 1:].T, delta2) * self.derivada_ativacao(y1)
                
                # Atualizacao
                self.W2 += self.eta * np.dot(delta2, y1_bias.T)
                self.W1 += self.eta * np.dot(delta1, x_k.T)
                
            mse_atual = erro_epoca / N
            self.learning_curve.append(mse_atual)
            
            if abs(erro_anterior - mse_atual) < self.precision:
                break
                
            erro_anterior = mse_atual
            epochs += 1
            
        return self

    def predict(self, X_test):
        v1 = np.dot(self.W1, X_test)
        y1 = self.ativacao(v1)
        
        bias_oculto = np.ones((1, X_test.shape[1]))
        y1_bias = np.vstack((bias_oculto, y1))
        
        v2 = np.dot(self.W2, y1_bias)
        y2 = self.ativacao(v2)
        
        return np.argmax(y2, axis=0)
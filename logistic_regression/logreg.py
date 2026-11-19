import numpy as numpy

class Logreg:
    def __init__(self, alpha=0.01, epochs=1000, optimizer="batch", batch_size=32):
        self.alpha = alpha
        self.epochs = epochs
        self.optimizer = optimizer.lower()
        self.batch_size = batch_size
        self.weights = None

    # _Mafonction est une convention pour les fonctions internes de la classe
    def _sigmoid(self, z):
        # clip permet de ne pas avoir de valeurs trop grandes qui pourrait amener a infini et donc des divisions par 0
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _update_weights(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.weights)
        predictions = 1 / (1 + np.exp(-z))
        gradient = np.dot(X.T, (predictions - y)) / m
        self.weights -= alpha * gradient

    def fit(self, X, y):
        # m = lignes de X
        # n_features = colonnes de X
        m, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        # Transforme y en un tableau (x, ) de la bonne taille (-1 = la fonction trouve la bonne taille elle meme)
        y = y.reshape(-1)

        # _ est une convention pour dire qu'on ne va pas utiliser la variable
        for _ in range(self.epochs):
            if self.optimizer == "batch":
                self._update_weights(X, y)
            elif self.optimizer == "sgd":
                indices = np.random.permutation(m)
                for i in indices:
                    # la forme i:i+1 permet de travailler avec un tableau 2D
                    # X[i] renverrait la ligne de notes de l'eleves comme un tableau 1D
                    # Dans _update_weights, X.shape[0] vaudrait alors le nombre de colonnes au lieu du nombre de lignes
                    # Cela nous permet de couper (slice) dans le tableau afin d'indiquer X.shape[0] = 1 (1 ligne du tableau) | X.shape[1] = n_features
                    _update_weights(X[i:i+1], y[i:i+1])
            elif self.optimizer == "mbsgd" or self.optimizer == "minibatch":
                indices = np.random.permutation(m)
                for start in range(0, m, self.batch_size):
                    # on part de start, pour faire une tranche jusqu'a start + batch size
                    idx = indices[start:start + self.batch_size]
                    self._update_weights(X[idx], y[idx])
            else:
                raise ValueError(f"Optimizer not supported : {self.optimizer}")

            return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights)
        return self._sigmoid(z)

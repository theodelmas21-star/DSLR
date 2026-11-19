import numpy as np

class Logreg:
    def __init__(self, alpha=0.01, epochs=1000, optimizer="batch", batch_size=32):
        self.alpha = alpha
        self.epochs = epochs
        self.optimizer = optimizer.lower()
        self.batch_size = batch_size
        self.weights = None
        self.loss_history = []

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
        self.weights -= self.alpha * gradient

    def _compute_loss(X, y):
        # On maintient la proba entre 0.0000...01 (1e-15) et 0.9999...99 (1 - 1e-15)
        # On evite ainsi de diviser par 0 si le modele est sur a 100%
        y_pred = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        
        y_vec = y.reshape(-1)

        # loss correspond ici a la Binary Cross-Entropy, la moyenne de penalite d'erreur globale
        # Elle nous permet de voir quand les resultats convergent, on peut donc adapter les parametres plus precisement
        # Le resultat de la formule est negatif, donc on fait -np.mean pour avoir le resultat positif
        # Plus le res est proche de 0, moins il y a d'erreur
        # Plus le modele est confiant et se trompe, plus il sera penalise
        # Si l'eleve est dans la maison, y_vec vaut 1, donc la formule renvoie np.log(y_pred)
        # Si l'eleve n'est pas dans la maison, y_vec vaut 0, on obtient np.log(1 - y_pred)
        loss = -np.mean(
            y_vec * np.log(y_pred) + (1 - y_vec) * np.log(1 - y_pred)
        )

        # loss est ici un scalaire Numpy, on le renvoie en float afin de pouvoir l'utiliser plus tard
        # Par exemple: si on veut serialiser en JSON, il leverait une exception face a un type scalaire NumPy
        return float(loss)

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
                    self._update_weights(X[i:i+1], y[i:i+1])
            elif self.optimizer == "mbsgd" or self.optimizer == "minibatch":
                indices = np.random.permutation(m)
                for start in range(0, m, self.batch_size):
                    # on part de start, pour faire une tranche jusqu'a start + batch size
                    idx = indices[start:start + self.batch_size]
                    self._update_weights(X[idx], y[idx])
            else:
                raise ValueError(f"Optimizer not supported : {self.optimizer}")

                self.loss_history.append(self._compute_loss(X, y))

            return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights)
        return self._sigmoid(z)

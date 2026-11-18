#astronomy #herbology #divination #muggle #ancient #history #transfig #charm #flying

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import sys

if len(sys.argv) != 2 or sys.argv[1] != "datasets/dataset_train.csv":
    print("Erreur : Fichier invalide ou manquant.")
    print("Usage attendu : python logreg_train.py datasets/dataset_train.csv")
    sys.exit(1)

train_path = sys.argv[1]

df = pd.read_csv(train_path)

house_col = "Hogwarts House"

choosen_cols = [
    "Astronomy",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Charms",
    "Flying"
]

houses = [
    "Gryffindor",
    "Slytherin",
    "Ravenclaw",
    "Hufflepuff"
]

for col in choosen_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    
df[choosen_cols] = df[choosen_cols].fillna(df[choosen_cols].mean())

# Means et stds correspondent a des Series : des objets Pandas correspondant a un dict
# Les cles sont les noms des matieres et les valeurs sont les moyennees et stds
means = df[choosen_cols].mean()
stds = df[choosen_cols].std()

# Normalisation par score Z, : res x stds = note - moyenne
df[choosen_cols] = (df[choosen_cols] - means) / stds

X_features = df[choosen_cols].values
m = len(X_features)
bias_column = np.ones((m, 1))
X = np.hstack((bias_column, X_features))

alpha = 0.01
epochs = 1000
m = len(X)

weights_dict = {}

for house in houses:
    # Creer un tableau numpy, -1 = il mettra autant de lignes que necessaire
    # 1 = une colonne
    y = (df[house_col] == house).astype(int).values.reshape(-1, 1)

    # Tableau des poids, autant de lignes que de matieres +1 pour le biais
    W = np.zeros((len(choosen_cols) + 1, 1))

    for epoch in range(epochs):
        # Z est le tableau qui contient le resultat des notes x leurs poids
        z = np.dot(X,W)

        # L'exponentielle permet d'avoir un resultat positif, -z permet de l'inverser
        # Un bon score aura une note tres basse dans les negatifs, son exponentielle sera donc proche de 0 et inversement
        # Cela nous donne le pourcentage de chance que l'eleve soit dans la maison
        predictions = 1 / (1 + np.exp(-z))

        # Gradient est le tableau de X transposee x l'erreur / len(X)
        # X transposee est X avec les colonnes a la place des lignes et inversement
        # Cela fait qu on multiplie l'erreur pour chaque matiere plutot que pour chaque eleve
        # On divise ensuite par la taille d'eleves pour ne pas avoir un nombre trop grand
        gradient = np.dot(X.T, (predictions - y)) / m

        # On modifie le tableau de poids, on soustrait le mouvement qu'on souhaite (alpha * gradient)
        # Pour que W partent dans le sens inverse, si l'erreur est positive alors le poids doit baisser et inversement
        W = W - alpha * gradient
        
    weights_dict[house] = W


# Envoie du Resultat dans un fichier JSON

model_data = {
    "means": means.to_dict(),
    "stds": stds.to_dict(),
    "weights": {}
}

# W etant un tableau NumPy (matrice de taille (len(choosen_cols) + 1, 1))
# Il n'est pas pris en compte par JSON, on l'applatit donc en une ligne avec flatten,
# Un tableau de taille (len(choosen_cols) + 1, ), puis on le passe en liste
for house, W in weights_dict.items():
    model_data["weights"][house] = W.flatten().tolist()

with open("logistic_regression/weights.json", "w") as f:
    json.dump(model_data, f, indent=4)
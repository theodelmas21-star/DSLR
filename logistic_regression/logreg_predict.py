import json
import pandas as pd
import numpy as np

df_test = pd.read_csv("datasets/dataset_test.csv")

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

df_test[choosen_cols] = df_test[choosen_cols].fillna(df_test[choosen_cols].mean())

with open("logistic_regression/weights.json", "r") as f:
    data = json.load(f)

means_dict = data["means"]
stds_dict = data["stds"]

for col in choosen_cols:
    df_test[col] = (df_test[col] - means_dict[col]) / stds_dict[col]

df_test.insert(0, "bias", 1)
cols_with_bias = ["bias"] + choosen_cols

X_test = df_test[cols_with_bias].values

probabilities_dict = {}

for house, weights_list in data["weights"].items():
    # Converti la liste de poids en tableau NumPy
    W = np.array(weights_list).reshape(-1, 1)

    # Calcul des scores bruts z = X * W
    z = np.dot(X_test, W)

    # Application de la fonction sigmoide (proba entre 0 et 1)
    predictions = 1 / (1 + np.exp(-z))

    probabilities_dict[house] = predictions.flatten()

# Creation d'un tableau a 4 colonnes pour les maisons avec les proba de chaque eleve en lignes
houses_list = list(data["weights"].keys())
proba_matrix = np.column_stack([probabilities_dict[h] for h in houses_list])

# Un tableau (x, ), contenant a chaque index l'indice de la maison la plus probable
best_indices = np.argmax(proba_matrix, axis=1)

# On transforme les index en noms reels de maisons 
final_predictions = [houses_list[i] for i in best_indices]
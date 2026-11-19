import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from logreg import Logreg

df = pd.read_csv("datasets/dataset_train.csv")

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

means = df[choosen_cols].mean()
stds = df[choosen_cols].std()

df[choosen_cols] = (df[choosen_cols] - means) / stds

X_features = df[choosen_cols].values
m = len(X_features)
bias_column = np.ones((m, 1))
X = np.hstack((bias_column, X_features))

models = {}

for house in houses:
    y = (df[house_col] == house).astype(int).values.reshape(-1, 1)

    clf = Logreg(alpha=0.1, epochs=1000, optimizer="batch")
    clf.fit(X, y)
    models[house] = clf

probas_list = [models[house].predict_proba(X) for house in houses]
probas_matrix = np.column_stack(probas_list)
best_house_indices = np.argmax(probas_matrix, axis=1)
final_predictions = [houses[idx] for idx in best_house_indices]

output_df = pd.DataFrame({
    "Index": df["Index"],
    "Hogwarts House": final_predictions
})

output_df.to_csv("logistic_regression/batch.csv", index=False)

y_true = df[house_col].to_numpy()
accuracy = np.mean(np.array(final_predictions) == y_true)
print(f"Accuracy : {accuracy:.2%}")
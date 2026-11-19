import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from logreg import Logreg

def logreg_train_predict(falpha=0.01, fepochs=1000, foptimizer="batch", fbatch_size=32, filename="logistic_regression/batch.csv"):
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
    loss_histories = {}

    for house in houses:
        y = (df[house_col] == house).astype(int).values.reshape(-1, 1)

        clf = Logreg(alpha=falpha, epochs=fepochs, optimizer=foptimizer, batch_size=fbatch_size)
        clf.fit(X, y)
        models[house] = clf
        loss_histories[house] = clf.loss_history

    probas_list = [models[house].predict_proba(X) for house in houses]
    probas_matrix = np.column_stack(probas_list)
    best_house_indices = np.argmax(probas_matrix, axis=1)
    final_predictions = [houses[idx] for idx in best_house_indices]

    output_df = pd.DataFrame({
        "Index": df["Index"],
        "Hogwarts House": final_predictions
    })

    output_df.to_csv(filename, index=False)

    y_true = df[house_col].to_numpy()
    accuracy = np.mean(np.array(final_predictions) == y_true)
    print(f"Accuracy : {accuracy:.2%}")
    print()

    return loss_histories

batch_loss = logreg_train_predict(foptimizer="batch")
minibatch_loss = logreg_train_predict(falpha=0.05, fepochs=200, foptimizer="minibatch", fbatch_size=32, filename="logistic_regression/minibatch.csv")
sgd_loss = logreg_train_predict(falpha=0.01, fepochs=60, foptimizer="sgd", filename="logistic_regression/sgd.csv")

target_house = "Gryffindor"

plt.figure(figsize=(8,4))
plt.plot(batch_loss[target_house], label="batch(1000 epochs)")
plt.plot(minibatch_loss[target_house], label="minibatch(200 epochs)")
plt.plot(sgd_loss[target_house], label="sgd(60 epochs)")

plt.title(f"Evolution de la loss - {target_house}")
plt.xlabel("Epoque")
plt.ylabel("Binary Cross-Entropy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
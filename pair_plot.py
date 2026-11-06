import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("datasets/dataset_train.csv")

house_col = "Hogwarts House"

numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
if "Index" in numeric_cols:
    numeric_cols.remove("Index")

    REGARDE TUTO SEABORN

sns.pairplot(df, hue="Hogwarts House")

plt.show()
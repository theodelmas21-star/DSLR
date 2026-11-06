import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("datasets/dataset_train.csv")

house_col = "Hogwarts House"

numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
if "Index" in numeric_cols:
    numeric_cols.remove("Index")

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Take the absolute values of the correlation matrix to find the highest with idxmax
corr_matrix = df[numeric_cols].corr()
corr_abs = corr_matrix.abs()
for col in numeric_cols:
  corr_abs.loc[col, col] = float("nan")


max_corr_pair = corr_abs.stack().idxmax()
max_corr_value = corr_abs.loc[max_corr_pair[0], max_corr_pair[1]]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x = max_corr_pair[0],
    y = max_corr_pair[1],
    hue="Hogwarts House",
    palette="coolwarm",
    alpha=0.8
)

plt.xlabel(f"{max_corr_pair[0]}")
plt.ylabel(f"{max_corr_pair[1]}")
plt.legend(title="Maisons")
plt.tight_layout()

plt.show()
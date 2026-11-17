import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("datasets/dataset_train.csv")

house_col = "Hogwarts."

df.columns = [
    col if len(col) <= 10 else col[:8] + '.' for col in df.columns
]

numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
if "Index" in numeric_cols:
    numeric_cols.remove("Index")

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

numeric_cols.append(house_col)

g = sns.pairplot(
    df[numeric_cols],
    hue="Hogwarts.",
    height=1.8,
    plot_kws={"s": 10, "alpha": 0.7}
)

g.figure.suptitle(f"Correlation between courses", fontsize=16, y=1.02)

sns.move_legend(g, "center right", bbox_to_anchor=(1.01, 0.5), frameon=True, title="Hogwarts Houses")

for ax in g.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=8, rotation=0, ha="right", va="center")

g.savefig(f"Correlation between courses", dpi=150, bbox_inches="tight")
plt.close("all")
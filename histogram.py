import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("datasets/dataset_train.csv")

house_col = "Hogwarts House"

numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
if "Index" in numeric_cols:
    numeric_cols.remove("Index")

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    min_val = df[col].min()
    max_val = df[col].max()
    if not max_val - min_val == 0:
        df[col] = (df[col] - min_val) / (max_val - min_val)

df_melted = df.melt(
    id_vars=[house_col],
    value_vars=numeric_cols,
    var_name="Course",
    value_name="Score",
)

plt.figure(figsize=(16, 8))

sns.boxplot(
    x="Course", y="Score", hue=house_col, data=df_melted, palette="Set2"
)

plt.xticks(rotation=45, ha="right")
plt.title(
    "Comparison of score distribution by course and by Hogwarts House"
)
plt.xlabel("Courses")
plt.ylabel("Scores")
plt.legend(title="Maisons")
plt.tight_layout()

plt.show()
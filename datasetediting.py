from datasets import load_dataset
import pandas as pd

dataset = load_dataset("UniversalCEFR/cefr_sp_en", split="train")

df = pd.DataFrame({
    "text": dataset["text"],
    "level": dataset["cefr_level"]
})
df = df[df["text"].str.strip() != ""]

max_size = df["level"].value_counts().max()

balanced_df = (
    df.groupby("level", group_keys=False)
      .apply(lambda x: x.sample(max_size, replace=True, random_state=42))
)

balanced_df.to_csv("balanced_cefr.csv", index=False)
print("Размер датасета:", len(balanced_df))
print("\nРаспределение классов:")
print(balanced_df["level"].value_counts())

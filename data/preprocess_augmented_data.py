import pandas as pd


def preprocess(m_df):
    m_df = m_df.sample(frac=1).reset_index(drop=True)
    m_df = m_df.replace(["\(", "\)"], [" ", " "], regex=True)
    m_df = m_df.replace(["<NULL>"], [""], regex=True)

    print("Preprocessing done.")

    return m_df


df = pd.read_csv("augmented/test.csv", sep=",").head(8)
df = preprocess(df)
df.to_csv("augmented/test.tsv", index=False, sep="\t")

df = pd.read_csv("augmented/train.csv", sep=",")
df = preprocess(df)
df.to_csv("augmented/train.tsv", index=False, sep="\t")

df = df.sample(frac=0.5).reset_index(drop=True)
df.to_csv("augmented/half_train.tsv", index=False, sep="\t")

df = pd.read_csv("augmented/valid.csv", sep=",")
df = preprocess(df)
df.to_csv("augmented/valid.tsv", index=False, sep="\t")

df = df.head(50_000)
df.to_csv("augmented/valid_mini.tsv", index=False, sep="\t")

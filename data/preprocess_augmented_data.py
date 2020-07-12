import pandas as pd


def preprocess(m_df):
    m_df = m_df.sample(frac=1).reset_index(drop=True)
    m_df = m_df.replace(["\(", "\)"], [" ", " "], regex=True)
    m_df = m_df.replace(["<NULL>"], [""], regex=True)

    print("Preprocessing done.")

    return m_df


df = pd.read_csv("augmented/train.csv", sep=",").head(361600)
df = preprocess(df)
df.to_csv("augmented/train.tsv", index=False, sep="\t")

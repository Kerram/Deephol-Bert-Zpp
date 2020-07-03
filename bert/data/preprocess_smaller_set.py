import pandas as pd


def preprocess(m_df):
    m_df = m_df.sample(frac=0.5).reset_index(drop=True)
    m_df = m_df.replace(['\(', '\)'], [' ', ' '], regex=True)
    m_df = m_df.replace(['<NULL>'], [''], regex=True)

    print("Preprocessing done.")

    return m_df


df = pd.read_csv('train.csv', sep=',')
df = preprocess(df)
df.to_csv('preprocessed_train_half.tsv', index=False, sep='\t')

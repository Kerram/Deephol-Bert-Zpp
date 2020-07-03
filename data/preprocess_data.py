import pandas as pd


def preprocess(m_df):
    m_df = m_df.sample(frac=1).reset_index(drop=True)
    m_df = m_df.replace(['\(', '\)'], [' ', ' '], regex=True)
    m_df = m_df.replace(['<NULL>'], [''], regex=True)

    print("Preprocessing done.")

    return m_df


df = pd.read_csv('test.csv', sep=',').head(8)
df = preprocess(df)
df.to_csv('preprocessed_test.tsv', index=False, sep='\t')

df = pd.read_csv('train.csv', sep=',')
df = preprocess(df)
df.to_csv('preprocessed_train.tsv', index=False, sep='\t')

df = pd.read_csv('valid.csv', sep=',')
df = preprocess(df)
df.to_csv('preprocessed_valid.tsv', index=False, sep='\t')

df = df.head(50_000)
df.to_csv('preprocessed_valid_mini.tsv', index=False, sep='\t')

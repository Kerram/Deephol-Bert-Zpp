import pandas as pd

train_df = pd.read_csv('train.csv', sep=',')
train_df = train_df.sample(50_000)
train_df.to_csv('train.tsv', index=False, sep='\t')

valid_df = pd.read_csv('valid.csv', sep=',')
valid_df = valid_df.sample(50_000)
valid_df.to_csv('valid.tsv', index=False, sep='\t')

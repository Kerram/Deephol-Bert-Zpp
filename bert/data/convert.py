import pandas as pd

test_df = pd.read_csv('mini-train.csv', sep=',').head(1)
test_df.to_csv('test.tsv', index=False, sep='\t')

train_df = pd.read_csv('mini-train.csv', sep=',')
train_df.to_csv('train.tsv', index=False, sep='\t')

valid_df = pd.read_csv('mini-valid.csv', sep=',')
valid_df.to_csv('valid.tsv', index=False, sep='\t')

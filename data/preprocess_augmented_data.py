import pandas as pd


# Enhance function by the courtesy of Filip Mikina.
def enhance(df):
    last_row_ind = 0
    seen_false = False
    seen_true = False
    for ind, row in df.iterrows():
        if ind % 10000 == 0:
            print("DONE", ind)

        if not row['is_negative']:
            if seen_true and seen_false:
                seen_false = False
                seen_true = False

                if (ind - last_row_ind) % 8 != 0:
                    print("ERROR", ind)

                df.loc[last_row_ind:(ind - 1), 'weight'] = 1 / ((ind - last_row_ind) // 8)
                last_row_ind = ind

            seen_false = True
        else:
            seen_true = True

    # Update last
    ind = len(df)

    if (ind - last_row_ind) % 8 != 0:
        print("ERROR", ind)

    df.loc[last_row_ind:(ind - 1), 'weight'] = 1 / ((ind - last_row_ind) // 8)
    return df


def preprocess(m_df):
    m_df = m_df.sample(frac=1).reset_index(drop=True)
    m_df = m_df.replace(["\(", "\)"], [" ", " "], regex=True)
    m_df = m_df.replace(["<NULL>"], [""], regex=True)

    print("Preprocessing done.")

    return m_df


df = pd.read_csv("augmented/test.csv", sep=",").head(8)
df = enhance(df)
df = preprocess(df)
df.to_csv("augmented/test.tsv", index=False, sep="\t")

df = pd.read_csv("augmented/train.csv", sep=",")
df = enhance(df)
df = preprocess(df)
df.to_csv("augmented/train.tsv", index=False, sep="\t")

df = df.sample(frac=0.5).reset_index(drop=True)
df.to_csv("augmented/half_train.tsv", index=False, sep="\t")

df = pd.read_csv("augmented/valid.csv", sep=",")
df = enhance(df)
df = preprocess(df)
df.to_csv("augmented/valid.tsv", index=False, sep="\t")

df = df.head(50_000)
df.to_csv("augmented/valid_mini.tsv", index=False, sep="\t")

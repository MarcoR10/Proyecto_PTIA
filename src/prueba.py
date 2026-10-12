import pandas as pd
df = pd.read_csv("../data/water.txt", parse_dates=["timestamp"])
print(df.shape)
print(df.columns[:10].tolist())
print(df.drop(columns="timestamp").isna().mean().describe())
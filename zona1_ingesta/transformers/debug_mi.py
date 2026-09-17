import pandas as pd
df = pd.read_csv("zona1_ingesta/data/raw/credit-scoring.csv")
print(pd.crosstab(df["Marital"], df["Sex"]))

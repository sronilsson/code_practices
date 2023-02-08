import pandas as pd
import numpy as np

LEN = 1000
TRAIN_V_TEST = "test"

df = pd.DataFrame(index=list(range(1, LEN)))

state_abbr = list(
    pd.read_csv(
        "/Users/simon/Desktop/envs/GEICO/lookups/us_state_abbr.csv", header=None
    )[0]
)


df["Feature_1"] = np.random.choice(["A", "B", "C", "D"], len(df))
df["Feature_2"] = np.random.choice(list(range(18, 101)), len(df))
df["Feature_3"] = np.random.choice(["low", "medium", "moderate", "high"], len(df))
df["Feature_4"] = np.random.choice(list(range(0, 10)), len(df))
df["Feature_5"] = np.random.choice(state_abbr, len(df))
if TRAIN_V_TEST == "train":
    df["Target"] = np.random.choice(list(range(0, 100)), len(df))
for col in df.columns:
    df.loc[df.sample(n=4).index, col] = pd.NA

df.to_csv("data/data_{}_{}.csv".format(str(LEN), TRAIN_V_TEST))

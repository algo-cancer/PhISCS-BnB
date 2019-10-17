import pandas as pd


def function(x):
    i = x.query('method == "BnB_DynamicMWMBounding_None_False"')
    j = x.query('method == "PhISCS_I_"')
    print(x)
    if not (i.empty or j.empty) and (i.runtime < j.runtime):
        return x


file = "../reports/10,10_10-16-04-17-17.csv"
df = pd.read_csv(file)
print(df.groupby("hash", group_keys=False).apply(function))

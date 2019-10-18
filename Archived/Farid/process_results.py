import pandas as pd
from glob import glob
from collections import defaultdict

kk_all = defaultdict(lambda: 0)
kk_mwm = defaultdict(lambda: 0)

def get_counter(file):
    df = pd.read_csv(file)
    df = df[['hash', 'method', 'nf', 'runtime']]
    counter = 0
    for _, df_group in df.groupby("hash"):
        k = df_group['nf'].values[0]
        kk_all[k] += 1
        mydict = pd.Series(df_group.runtime.values, index=df_group.method).to_dict()
        if mydict['PhISCS_I_'] > mydict['BnB_DynamicMWMBounding_None_False']:
            counter += 1
            kk_mwm[k] += 1
    return counter

# file = "/data/frashidi/Phylogeny_BnB/reports/10,10_10-16-04-17-17.csv"
# get_counter(file)

for file in glob('/data/frashidi/Phylogeny_BnB/reports/*10-16-*'):
    get_counter(file)

print(sorted(kk_all.items() ,  key=lambda x: x[0]))
print()
print(sorted(kk_mwm.items() ,  key=lambda x: x[0]))

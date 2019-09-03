import numpy as np
from instances import I1
from funcs import *
import pybnb
from utils import *
import operator
from collections import defaultdict
import time
import pandas as pd
from tqdm import tqdm
import itertools
from BnB import *


if __name__ == '__main__':
    methods = [
        "PhISCS_a_True",
        "PhISCS_a_False",
        "PhISCS_b_True",
        "PhISCS_b_False",
        "PhISCS_I",
    ]
    df = pd.DataFrame()
    # n: number of Cells
    # m: number of Mutations
    for n, m in tqdm(itertools.product([5, 6, 7, 8, 9 ], repeat=2)):
        for i in range(5):
            x = np.random.randint(2, size=(n, m))
            for method in methods:
                ans, nf, runTime = solveWith(method, x)
                row = {
                    "n": str(n),
                    "m": str(m),
                    "hash": hash(x.tostring()),
                    "method": method,
                    "runtime": str(runTime),
                    "nf": str(nf),
                }
                # print(row)
                df = df.append(row, ignore_index=True)
    print(df)
    df.to_csv(f"report_{df.shape}_{time.time()}.csv")



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
import itertools, os
from BnB import *
from boundingAlgs import *
from lp_bounding import LP_Bounding, LP_Bounding_direct, LP_Bounding_direct_4

## still working copied from compareOPT

if __name__ == '__main__':
    scriptName = os.path.basename(__file__).split(".")[0]
    print(f"{scriptName} starts here")
    methods = [
        myPhISCS_I,
        randomPartitionBounding,
        greedyViolationsPartitionBounding,
        greedyPartitionBounding,
        mxWeightedMatchingPartitionBounding,
        LP_Bounding,
        LP_Bounding_direct,
        LP_Bounding_direct_4,

        # mxMatchingPartitionBounding,
    ]
    df = pd.DataFrame(columns=["hash", "n",	"m", "nf",	"method", "runtime"])
    # n: number of Cells
    # m: number of Mutations
    iterList = itertools.product([12,], # n
                                 [12,], # m
                                 list(range(1)) # i
                                 )
    iterList = list(iterList)
    for n, m, ind in tqdm(iterList):
            x = np.random.randint(2, size=(n, m))
            for method in methods:
                runTime = time.time()
                nf = method(x)
                runTime = time.time() - runTime
                row = {
                    "n": str(n),
                    "m": str(m),
                    "hash": hash(x.tostring()),
                    "method": f"{method.__name__ }",
                    "runtime": str(runTime)[:6],
                    "nf": str(nf),
                    # "desc": desc
                }
                # print(row)
                df = df.append(row, ignore_index=True)
    print(df)
    # csvFileName = f"report_{scriptName}_{df.shape}_{time.time()}.csv"
    # df.to_csv(csvFileName)
    # print(f"CSV file stored at {csvFileName}")



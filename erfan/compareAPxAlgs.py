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

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

def myPhISCS_I(x):
    solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1) = PhISCS_I(x, beta=0.99, alpha=0.00000001)
    nf = len(np.where(solution != x)[0])
    return nf

def getKPartitionedPhISCS(k):
    @rename(f'partitionedPhISCS_{k}')
    def partitionedPhISCS(x):
        ans = 0
        for i in range(x.shape[1]//k):
            ans += myPhISCS_I(x[:, i * k: (i+1) * k])
        if x.shape[1] % k >= 2:
            ans += myPhISCS_I(x[:, ((x.shape[1]//k) * k): ])
        return ans
    return partitionedPhISCS


if __name__ == '__main__':
    scriptName = os.path.basename(__file__).split(".")[0]
    print(f"{scriptName} starts here")
    methods = [
        myPhISCS_I,
        # getKPartitionedPhISCS(2),
        # getKPartitionedPhISCS(3),
        # getKPartitionedPhISCS(4),
        getKPartitionedPhISCS(5),
        # randomPartitionBounding,
        # greedyPartitionBounding,
        mxWeightedMatchingPartitionBounding,
        LP_Bounding,
        LP_Bounding_direct,
        LP_Bounding_direct_4,

        # mxMatchingPartitionBounding,
    ]
    df = pd.DataFrame(columns=["hash", "n",	"m", "nf",	"method", "runtime"])
    # n: number of Cells
    # m: number of Mutations
    iterList = itertools.product([8, 20], # n
                                 [15, 20], # m
                                 list(range(5)) # i
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
                    "runtime": str(runTime),
                    "nf": str(nf),
                    # "desc": desc
                }
                # print(row)
                df = df.append(row, ignore_index=True)
    print(df)
    csvFileName = f"report_{scriptName}_{df.shape}_{time.time()}.csv"
    df.to_csv(csvFileName)
    print(f"CSV file stored at {csvFileName}")



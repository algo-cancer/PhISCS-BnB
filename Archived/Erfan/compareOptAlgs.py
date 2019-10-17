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
from boundingAlgs import *
from ips import ILP_1, LP_2
from lp_bounding import LP_Bounding, LP_Bounding_direct
from erfan_BnB import ErfanBnBSolver, ErfanBnB
import phylogeny_lb
import copy


def solveWith(name, bounding, x):
    ans = copy.copy(x)
    runTime = time.time()
    if isinstance(name, str) and "PhISCS_a" in name:
        assert False
        problem = PhISCS_a(x)
        results = pybnb.solve(problem, log=None)
        ans = results.best_node.state[0]
        desc = f"{results.nodes}"
    elif isinstance(name, str) and "PhISCS_b" in name:
        assert False
        problem = PhISCS_b(x)
        results = pybnb.solve(problem, log=None)
        ans = results.best_node.state[0]
        desc = f"{results.nodes}"
    elif  isinstance(name, str) and "PhISCS_c" in name:
        problem = PhISCS_c(x, bounding)
        results = pybnb.solve(problem, log=None)
        # results = pybnb.solve(problem)
        ans = results.best_node.state[0]
        desc = f"{results.nodes}"
    elif isinstance(name, str) and "Sept24_BnB" in name:
        problem = Sept24_BnB(x, bounding)
        results = pybnb.solve(problem, log=None)
        # results = pybnb.solve(problem)
        flipList = results.best_node.state[0]
        for a, b in flipList:
            ans[a, b] = 1
        desc = f"{results.nodes}"
    elif isinstance(name, str) and "ErfanBnB" in name:
        problem = ErfanBnB(x, bounding)
        results = pybnb.solve(problem, log=None)
        ans = results.best_node.state[0]
        desc = f"{results.nodes}"
    elif isinstance(name, str) and "PhISCS_I" in name:
        solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1) = PhISCS_I(x, beta=0.99, alpha=0.00000001)
        ans = solution
        desc = ""
    elif callable(name):
        ans = name(x)
        desc = ""
    else:
        print(f"Method {name} does not exist.")

    runTime = time.time() - runTime
    nf = len(np.where(ans != x)[0])
    return ans, nf, runTime, desc



if __name__ == '__main__':
    scriptName = os.path.basename(__file__).split(".")[0]
    print(f"{scriptName} starts here")
    methods = [
      # "PhISCS_a",
      # "PhISCS_a_False",
      # "PhISCS_b",
      # ("PhISCS_c", randomPartitionBounding),
      # ("PhISCS_c", greedyPartitionBounding),
      ("PhISCS_c", mxWeightedMatchingPartitionBounding),
      # ("PhISCS_c", mxMatchingPartitionBounding),
      # ("PhISCS_c", LP_Bounding),
      # ("PhISCS_c", LP_Bounding_direct),
      # ("PhISCS_I", None),
      # ("ErfanBnB", None), What is this?
      # (ILP_1, None),
      # (LP_2, None),
      # ("Sept24_BnB", phylogeny_lb.lb_gurobi),
      ("Sept24_BnB", phylogeny_lb.lb_max_weight_matching),
      (ILP_1, None),
    ]
    df = pd.DataFrame(columns=["hash", "n", "m", "nf", "method", "runtime", "desc"])
    # n: number of Cells
    # m: number of Mutations
    iterList = itertools.product([5, 6, 7, 8], # n
                                 [5, 6, 7, 8 ], # m
                                 list(range(2)) # i
                                 )
    iterList = list(iterList)

    for n, m, i in tqdm(iterList):
    # for n, m, i in iterList:
            x = np.random.randint(2, size=(n, m))
            # print(repr(x))
            for method, bounding in methods:
                ans, nf, runTime, desc = solveWith(method, bounding, x)
                # print(get)
                methodName = method if isinstance(method, str) else method.__name__
                row = {
                    "n": str(n),
                    "m": str(m),
                    "hash": hash(x.tostring()),
                    "method": f"{methodName}_{'' if bounding is None else bounding.__name__ }",
                    "runtime": str(runTime)[:8],
                    "nf": str(nf),
                    "desc": desc
                }
                # print(row)
                df = df.append(row, ignore_index=True)
    print(df)
    # csvFileName = f"report_{scriptName}_{df.shape}_{time.time()}.csv"
    # df.to_csv(csvFileName)
    # print(f"CSV file stored at {csvFileName}")



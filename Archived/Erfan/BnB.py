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
from boundingAlgs import *
import phylogeny_lb
import copy


def get_a_coflict(D, p, q):
    oneone = None
    zeroone = None
    onezero = None
    for r in range(D.shape[0]):
        if D[r, p] == 1 and D[r, q] == 1:
            oneone = r
        if D[r, p] == 0 and D[r, q] == 1:
            zeroone = r
        if D[r, p] == 1 and D[r, q] == 0:
            onezero = r
        if oneone != None and zeroone != None and onezero != None:
            return (p, q, oneone, zeroone, onezero)
    return None


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def apply_flips(I, F):
    for i, j in F:
        I[i, j] = 1
    return I


def deapply_flips(I, F):
    for i, j in F:
        I[i, j] = 0
    return I


class Sept24_BnB(pybnb.Problem):
    def __init__(self, I, bounding_alg):
        self.I = I
        self.bounding_alg = bounding_alg
        self.F = []
        self.nzero = len(np.where(self.I == 0)[0])
        # self.lb, self.G, self.best_pair = 0, 0, 0
        self.lb, self.G, self.best_pair = self.bounding_alg(self.I, None, None)
        self.nflip = 0

    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.lb == 0:
            return self.nflip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        return self.nflip + self.lb

    def save_state(self, node):
        node.state = (self.F, self.G, self.best_pair, self.lb, self.nflip)

    def load_state(self, node):
        self.F, self.G, self.best_pair, self.lb, self.nflip = node.state

    def branch(self):
        p, q = self.best_pair
        I = apply_flips(self.I, self.F)
        p, q, oneone, zeroone, onezero = get_a_coflict(I, p, q)

        node_l = pybnb.Node()
        G = self.G.copy()
        F = self.F.copy()
        F.append((onezero, q))
        I[onezero, q] = 1
        new_lb, new_G, new_best_pair = self.bounding_alg(I, q, G)
        # new_lb, new_G, new_best_pair = 0, 0, 0
        node_l.state = (F, new_G, new_best_pair, new_lb, self.nflip + 1)
        node_l.queue_priority = -new_lb
        I[onezero, q] = 0

        node_r = pybnb.Node()
        G = self.G.copy()
        F = self.F.copy()
        F.append((zeroone, p))
        I[zeroone, p] = 1
        new_lb, new_G, new_best_pair = self.bounding_alg(I, p, G)
        # new_lb, new_G, new_best_pair = 0, 0, 0
        node_r.state = (F, new_G, new_best_pair, new_lb, self.nflip + 1)
        node_r.queue_priority = -new_lb

        self.I = deapply_flips(I, F)
        return [node_l, node_r]


class PhISCS_c(pybnb.Problem):
    def __init__(self, I, boundingAlg):
        self.I = I
        self.nflip = 0
        self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        self.boundVal = 0
        self.boundingAlg = boundingAlg

    def sense(self):
        return pybnb.minimize

    def objective(self):
        # print("Obj: ", self.icf, self.colPair)
        if self.icf:
            return self.nflip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        newBound = self.nflip + self.boundingAlg(self.I)
        self.boundVal = max(self.boundVal, newBound)
        return self.boundVal

    def save_state(self, node):
        node.state = (self.I, self.icf, self.colPair, self.boundVal, self.nflip)

    def load_state(self, node):
        self.I, self.icf, self.colPair, self.boundVal, self.nflip = node.state

    def branch(self):
        if self.icf:
            return
        p, q = self.colPair
        p, q, oneone, zeroone, onezero = get_a_coflict(self.I, p, q)

        node = pybnb.Node()
        I = self.I.copy()
        I[onezero, q] = 1
        newIcf, newColPar = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        node.state = (I, newIcf, newColPar, self.boundVal, self.nflip + 1)
        yield node

        node = pybnb.Node()
        I = self.I.copy()
        I[zeroone, p] = 1
        newIcf, newColPar = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        node.state = (I, newIcf, newColPar, self.boundVal, self.nflip + 1)
        yield node



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
    x = np.random.randint(2, size=(7, 7))
    ans, nf, runTime, desc = solveWith("PhISCS_c", randomPartitionBounding, x)
    print("ans = ", ans)
    print("nf = ", nf)
    print("runTime = ", runTime)
    print("desc = ", desc)
    ans, nf, runTime, desc = solveWith("Sept24_BnB", phylogeny_lb.lb_gurobi, x)
    print("ans = ", ans)
    print("nf = ", nf)
    print("runTime = ", runTime)
    print("desc = ", desc)
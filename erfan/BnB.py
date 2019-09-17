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

randomPartition = False

class PhISCS_b(pybnb.Problem):
    def __init__(self, I):
        self.I = I
        self.nflip = 0

    def sense(self):
        return pybnb.minimize

    def objective(self):
        icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        if icf:
            return self.nflip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        return self.nflip + get_lower_bound(self.I, partition_randomly=randomPartition)

    def save_state(self, node):
        node.state = (self.I, self.nflip)

    def load_state(self, node):
        self.I, self.nflip = node.state

    def branch(self):
        icf, (p, q) = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        p, q, oneone, zeroone, onezero = get_a_coflict(self.I, p, q)

        node = pybnb.Node()
        I = self.I.copy()
        I[onezero, q] = 1
        node.state = (I, self.nflip + 1)
        yield node

        node = pybnb.Node()
        I = self.I.copy()
        I[zeroone, p] = 1
        node.state = (I, self.nflip + 1)
        yield node


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
        # print("Branch: ", self.icf, self.colPair)
        # icf, (p,q) = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
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


def solveWith(name, x):
    randPart = "True" in name
    runTime = time.time()
    if "PhISCS_a" in name:
        problem = PhISCS_a(x)
        results = pybnb.solve(problem, log=None)
        ans = results.best_node.state[0]
    elif "PhISCS_b" in name:
        problem = PhISCS_b(x)
        results = pybnb.solve(problem, log=None)
        ans = results.best_node.state[0]
    elif "PhISCS_c" in name:
        problem = PhISCS_b(x)
        results = pybnb.solve(problem, log=None)
        ans = results.best_node.state[0]

    elif "PhISCS_I" in name:
        solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1) = PhISCS_I(x, beta=0.99, alpha=0.00000001)
        ans = solution
    else:
        print(f"Method {name} does not exist.")

    runTime = time.time() - runTime
    nf = len(np.where(ans != x)[0])
    return ans, nf, runTime


if __name__ == '__main__':
    result = solveWith("PhISCS_c", np.zeros((5, 5)))
    print(result)
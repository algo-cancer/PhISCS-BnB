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


class PhISCS_a(pybnb.Problem):
    def __init__(self, I, partition_randomly=True):
        self.I = I
        self.X = np.where(self.I == 0)
        self.flip = 0
        self.idx = 0
        self.partition_randomly = partition_randomly

    def sense(self):
        return pybnb.minimize

    def objective(self):
        icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        if icf:
            return self.flip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        return self.flip + get_lower_bound(self.I, partition_randomly=self.partition_randomly)

    def save_state(self, node):
        node.state = (self.I, self.idx, self.flip)

    def load_state(self, node):
        self.I, self.idx, self.flip = node.state

    def branch(self):
        if self.idx < len(self.X[0]):
            node = pybnb.Node()
            I = self.I.copy()
            x = self.X[0][self.idx]
            y = self.X[1][self.idx]
            I[x, y] = 1
            node.state = (I, self.idx + 1, self.flip + 1)
            yield node

            node = pybnb.Node()
            I = self.I.copy()
            x = self.X[0][self.idx]
            y = self.X[1][self.idx]
            I[x, y] = 0
            node.state = (I, self.idx + 1, self.flip)
            yield node


class PhISCS_b(pybnb.Problem):
    def __init__(self, I, partition_randomly = True):
        self.I = I
        self.X = np.zeros(I.shape, dtype=bool)
        self.flip = 0
        self.partition_randomly = partition_randomly

    def sense(self):
        return pybnb.minimize

    def objective(self):
        icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        if icf:
            return self.flip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        return self.flip + get_lower_bound(self.I, partition_randomly=self.partition_randomly)

    def save_state(self, node):
        node.state = (self.I, self.flip)

    def load_state(self, node):
        self.I, self.flip = node.state

    def branch(self):
        icf, (p, q) = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        p, q, oneone, zeroone, onezero = get_a_coflict(self.I, p, q)

        if not self.X[onezero, q]:
            node = pybnb.Node()
            I = self.I.copy()
            I[onezero, q] = 1
            node.state = (I, self.flip + 1)
            yield node

        if not self.X[zeroone, p]:
            node = pybnb.Node()
            I = self.I.copy()
            I[zeroone, p] = 1
            node.state = (I, self.flip + 1)
            yield node


def solveWith(name, x):
    randPart = "True" in name
    runTime = time.time()
    if "PhISCS_a" in name:
        problem = PhISCS_a(x, randPart)
        results = pybnb.solve(problem, log=None)
        ans = results.best_node.state[0]
    elif "PhISCS_b" in name:
        problem = PhISCS_b(x, randPart)
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
    pass
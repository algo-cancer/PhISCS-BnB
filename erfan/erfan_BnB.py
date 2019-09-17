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
from lp_bounding import makeGurobiModel, flip, unFlipLast, LP_Bounding_Model, LP_brief


class ErfanBnB(pybnb.Problem):
    def __init__(self, I, boundingAlg):
        self.I = I
        self.nflip = 0
        self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        self.boundVal = 0
        self.boundingAlg = boundingAlg
        self.model, self.Y = makeGurobiModel(I)

    def sense(self):
        return pybnb.minimize

    def objective(self):
        # print("Obj: ", self.icf, self.colPair)
        if self.icf:
            return self.nflip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        t1 = time.time()
        newBound = LP_Bounding_Model(self.model)
        t2 = time.time()
        newBoundp = self.nflip + LP_brief(self.I)
        t3 = time.time()

        if newBound < newBoundp:
          print(repr(self.I))
          print(newBound, newBoundp)
          print(t2 - t1, t3 - t2)
          exit(0)
        # print()
        # print(newBound, newBoundp)
        # print(t2 - t1, t3 - t2)
        # print()
        # exit(0)
        self.boundVal = max(self.boundVal, newBound)
        return self.boundVal

    def save_state(self, node):
        node.state = (self.I, self.icf, self.colPair, self.boundVal, self.nflip, self.model, self.Y)

    def load_state(self, node):
        self.I, self.icf, self.colPair, self.boundVal, self.nflip, self.model, self.Y = node.state

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
        flip(onezero, q, self.model, self.Y)
        newIcf, newColPar = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        node.state = (I, newIcf, newColPar, self.boundVal, self.nflip + 1, self.model, self.Y)
        yield node
        unFlipLast(self.model)

        node = pybnb.Node()
        I = self.I.copy()
        I[zeroone, p] = 1
        flip(zeroone, q, self.model, self.Y)
        newIcf, newColPar = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        node.state = (I, newIcf, newColPar, self.boundVal, self.nflip + 1, self.model, self.Y)
        yield node
        unFlipLast(self.model)


def ErfanBnBSolver(x):
    problem = ErfanBnB(x, None)
    results = pybnb.solve(problem, log=None)
    ans = results.best_node.state[0]
    return ans

if __name__ == '__main__':
    n, m = 9, 9
    # x = np.random.randint(2, size=(n, m))
    x = np.array([[0, 0, 1, 1, 1, 1, 1],
       [1, 0, 1, 0, 1, 1, 1],
       [0, 1, 1, 0, 0, 1, 1],
       [0, 0, 1, 1, 1, 0, 1],
       [1, 0, 1, 1, 1, 1, 0],
       [0, 0, 1, 0, 0, 0, 1],
       [0, 1, 1, 0, 0, 0, 1]])

    xc = np.array([[0, 0, 1, 1, 1, 1, 1],
       [1, 0, 1, 0, 1, 1, 1],
       [0, 1, 1, 0, 0, 1, 1],
       [1, 0, 1, 1, 1, 0, 1],
       [1, 0, 1, 1, 1, 1, 0],
       [0, 0, 1, 0, 0, 0, 1],
       [0, 1, 1, 0, 0, 0, 1]])


    print(np.nonzero(x != xc))
    exit(0)
    print("HERE")
    problem = ErfanBnB(x, None)
    results = pybnb.solve(problem, log=None)
    ans = results.best_node.state[0]
    nf = len(np.where(ans != x)[0])
    print(x)
    print(ans)
    print(nf)


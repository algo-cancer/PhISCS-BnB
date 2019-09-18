import pybnb
import numpy as np
from utils import *
import operator
import time
from collections import defaultdict
from optparse import OptionParser
import networkx as nx
parser = OptionParser()
parser.add_option('-n', '--numberOfCells', dest='n', help='', type=int, default=5)
parser.add_option('-m', '--numberOfMutations', dest='m', help='', type=int, default=5)
options, args = parser.parse_args()


noisy = np.random.randint(2, size=(options.n, options.m))
# print(noisy)
noisy = np.array([
    [0,1,0,0,0,0,1,1,1,0],
    [0,1,1,0,1,1,1,0,1,0],
    [1,0,0,1,0,1,1,1,0,0],
    [1,0,0,0,0,0,0,1,0,0],
    [1,1,1,1,1,1,0,1,0,1],
    [0,1,1,1,1,1,1,1,0,0],
    [1,0,0,1,0,1,0,0,0,0],
    [1,1,1,1,0,0,1,0,1,1],
    [0,0,1,0,1,1,1,1,1,0],
    [1,1,1,1,0,0,1,0,1,1],
])
# ms_package_path = '/home/frashidi/software/bin/ms'
# ground, noisy, (countFN,countFP,countNA) = get_data(n=30, m=15, seed=1, fn=0.20, fp=0, na=0, ms_package_path=ms_package_path)
a = time.time()
solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1) = PhISCS_I(noisy, beta=0.9, alpha=0.00000001)
b = time.time()
print('PhISCS_I in seconds: {:.3f}'.format(b-a))
print('Number of flips reported by PhISCS_I:', len(np.where(solution != noisy)[0]))


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I):
    def sort_bin(a):
        b = np.transpose(a)
        b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
        idx = np.argsort(b_view.ravel())[::-1]
        c = b[idx]
        return np.transpose(c), idx

    O, idx = sort_bin(I)
    #todo: delete duplicate columns
    #print(O, '\n')
    Lij = np.zeros(O.shape, dtype=int)
    for i in range(O.shape[0]):
        maxK = 0
        for j in range(O.shape[1]):
            if O[i,j] == 1:
                Lij[i,j] = maxK
                maxK = j+1
    #print(Lij, '\n')
    Lj = np.amax(Lij, axis=0)
    #print(Lj, '\n')
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            if O[i,j] == 1:
                if Lij[i,j] != Lj[j]:
                    return False, (idx[j], idx[Lj[j]-1])
    return True, (None,None)


def get_a_coflict(D, p, q):
    #todo: oneone is not important you can get rid of
    oneone = None
    zeroone = None
    onezero = None
    for r in range(D.shape[0]):
        if D[r,p] == 1 and D[r,q] == 1:
            oneone = r
        if D[r,p] == 0 and D[r,q] == 1:
            zeroone = r
        if D[r,p] == 1 and D[r,q] == 0:
            onezero = r
        if oneone != None and zeroone != None and onezero != None:
            return (p,q,oneone,zeroone,onezero)
    return None


def all_possible_pairs_of_columns(lst):
    if len(lst) < 2:
        yield []
        return
    if len(lst) % 2 == 1:
        for i in range(len(lst)):
            for result in all_possible_pairs_of_columns(lst[:i] + lst[i+1:]):
                yield result
    else:
        a = lst[0]
        for i in range(1,len(lst)):
            pair = (a,lst[i])
            for rest in all_possible_pairs_of_columns(lst[1:i]+lst[i+1:]):
                yield [pair] + rest


def get_lower_bound(D, changed_column, previous_G):
    if changed_column == None:
        G = nx.Graph()
    else:
        G = previous_G

    def calc_min0110_for_one_pair_of_columns(p, q, G):
        foundOneOne = False
        numberOfZeroOne = 0
        numberOfOneZero = 0
        for r in range(D.shape[0]):
            if D[r,p] == 1 and D[r,q] == 1:
                foundOneOne = True
            if D[r,p] == 0 and D[r,q] == 1:
                numberOfZeroOne += 1
            if D[r,p] == 1 and D[r,q] == 0:
                numberOfOneZero += 1
        if foundOneOne:
            G.add_edge(p, q, weight=min(numberOfZeroOne, numberOfOneZero))
        else:
            G.add_edge(p, q, weight=0)

    if changed_column == None:
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                calc_min0110_for_one_pair_of_columns(p, q, G)
    else:
        q = changed_column
        for p in range(D.shape[1]):
            if p < q:
                calc_min0110_for_one_pair_of_columns(p, q, G)
            elif q < p:
                calc_min0110_for_one_pair_of_columns(q, p, G)

    best_pairing = nx.max_weight_matching(G)
    # print(best_pairing)
    lb = 0
    for a, b in best_pairing:
        lb += G[a][b]["weight"]
    return lb, G

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

best_objective = np.inf

class Phylogeny_BnB(pybnb.Problem):
    def __init__(self, I):
        self.I = I
        self.lb, self.G = get_lower_bound(self.I, None, None)
        self.icf, self.col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        self.nflip = 0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.nflip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        # if best_objective == 20:
        #     return 20
        # else:
        # lb, new_G = get_lower_bound(self.I, None, self.G)
        # self.lb = max(self.lb, self.nflip + lb)
        return self.lb
        # return 19

    def notify_new_best_node(self, node, current):
        # print('---------', node.objective)
        best_objective = node.objective

    def save_state(self, node):
        node.state = (self.I, self.G, self.icf, self.col_pair, self.lb, self.nflip)

    def load_state(self, node):
        self.I, self.G, self.icf, self.col_pair, self.lb, self.nflip = node.state

    def branch(self):
        if self.icf:
            return
        p, q = self.col_pair
        p,q,oneone,zeroone,onezero = get_a_coflict(self.I, p, q)
        
        node = pybnb.Node()
        I = self.I.copy()
        G = self.G.copy()
        I[onezero,q] = 1
        new_icf, new_col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        lb, new_G = get_lower_bound(I, q, G)
        self.lb = max(self.lb, self.nflip+1+lb)
        node.state = (I, new_G, new_icf, new_col_pair, self.lb, self.nflip+1)
        node.queue_priority = -1*self.lb
        yield node
        
        node = pybnb.Node()
        I = self.I.copy()
        G = self.G.copy()
        I[zeroone,p] = 1
        new_icf, new_col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        lb, new_G = get_lower_bound(I, p, G)
        self.lb = max(self.lb, self.nflip+1+lb)
        node.state = (I, new_G, new_icf, new_col_pair, self.lb, self.nflip+1)
        node.queue_priority = -1*self.lb
        yield node


problem = Phylogeny_BnB(noisy)
a = time.time()
results = pybnb.solve(problem, log_interval_seconds=10.0, queue_strategy='custom')
b = time.time()
print('Phylogeny_BnB in seconds: {:.3f}'.format(b-a))
if results.solution_status != 'unknown':
    print('Number of flips reported by Phylogeny_BnB:', results.best_node.state[-1])
    icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(results.best_node.state[0])
    print('Is the output matrix reported by Phylogeny_BnB conflict free:', icf)

import pybnb
import numpy as np
from utils import *
import operator
import time
from collections import defaultdict
from optparse import OptionParser
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


def get_lower_bound(D, appoc, changed_column, ifpoc_previous):
    importance_for_pairs_of_columns = defaultdict(lambda: 0)

    def calc_importance_for_one_pair_of_columns(p, q, importance_for_pairs_of_columns):
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
            importance_for_pairs_of_columns[(p,q)] = min(numberOfZeroOne, numberOfOneZero)
    
    if changed_column == None:
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                calc_importance_for_one_pair_of_columns(p, q, importance_for_pairs_of_columns)
    else:
        importance_for_pairs_of_columns = ifpoc_previous
        q = changed_column
        for p in range(D.shape[1]):
            if p < q:
                calc_importance_for_one_pair_of_columns(p, q, importance_for_pairs_of_columns)
            elif q < p:
                calc_importance_for_one_pair_of_columns(q, p, importance_for_pairs_of_columns)

    maximum_importance_for_pairs_of_columns = defaultdict(lambda: 0)
    mifpoc_index = {}
    index = 0
    for pairs in appoc:
        for pair in pairs:
            maximum_importance_for_pairs_of_columns[index] += importance_for_pairs_of_columns[pair]
        mifpoc_index[index] = pairs
        index += 1
    result = sorted(maximum_importance_for_pairs_of_columns.items(), key=operator.itemgetter(1), reverse=True)
    # print(mifpoc_index[result[0][0]])
    return result[0][1], importance_for_pairs_of_columns


appoc = list(all_possible_pairs_of_columns(list(range(noisy.shape[1]))))


# a = time.time()
# lb, ifpoc_previous = get_lower_bound(noisy, appoc, None, None)
# b = time.time()
# print('First: {:.5f}'.format(b-a))
# print(lb, ifpoc_previous)
# noisy = np.array([
#     [0,1,0,1,0,0,1,1,1,0],
#     [0,1,1,0,1,1,1,0,1,0],
#     [1,0,0,1,0,1,1,1,0,0],
#     [1,0,0,0,0,0,0,1,0,0],
#     [1,1,1,1,1,1,0,1,0,1],
#     [0,1,1,1,1,1,1,1,0,0],
#     [1,0,0,1,0,1,0,0,0,0],
#     [1,1,1,1,0,0,1,0,1,1],
#     [0,0,1,0,1,1,1,1,1,0],
#     [1,1,1,1,0,0,1,0,1,1]
# ])
# a = time.time()
# lb, ifpoc = get_lower_bound(noisy, appoc, 3, ifpoc_previous)
# b = time.time()
# print('Second: {:.5f}'.format(b-a))
# print(lb, ifpoc)
# exit()

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

best_objective = np.inf

class Phylogeny_BnB(pybnb.Problem):
    def __init__(self, I):
        self.I = I
        self.lb, self.ifpoc = get_lower_bound(self.I, appoc, None, None)
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
        lb, new_ifpoc = get_lower_bound(self.I, appoc, None, self.ifpoc)
        self.lb = max(self.lb, self.nflip + lb)
        return self.lb
        # return 19

    def notify_new_best_node(self, node, current):
        print('---------', node.objective)
        best_objective = node.objective

    def save_state(self, node):
        node.state = (self.I, self.ifpoc, self.icf, self.col_pair, self.lb, self.nflip)

    def load_state(self, node):
        self.I, self.ifpoc, self.icf, self.col_pair, self.lb, self.nflip = node.state

    def branch(self):
        if self.icf:
            return
        p, q = self.col_pair
        p,q,oneone,zeroone,onezero = get_a_coflict(self.I, p, q)
        
        node = pybnb.Node()
        I = self.I.copy()
        I[onezero,q] = 1
        new_icf, new_col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        new_ifpoc = None
        # lb, new_ifpoc = get_lower_bound(I, appoc, q, self.ifpoc)
        # self.lb = max(self.lb, self.nflip + lb)
        node.state = (I, new_ifpoc, new_icf, new_col_pair, self.lb, self.nflip+1)
        yield node
        
        node = pybnb.Node()
        I = self.I.copy()
        I[zeroone,p] = 1
        new_icf, new_col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        new_ifpoc = None
        # lb, new_ifpoc = get_lower_bound(I, appoc, p, self.ifpoc)
        # self.lb = max(self.lb, self.nflip + lb)
        node.state = (I, new_ifpoc, new_icf, new_col_pair, self.lb, self.nflip+1)
        yield node


problem = Phylogeny_BnB(noisy)
a = time.time()
results = pybnb.solve(problem, log_interval_seconds=10.0)#, queue_strategy='breadth')
b = time.time()
print('Phylogeny_BnB in seconds: {:.3f}'.format(b-a))
if results.solution_status != 'unknown':
    print('Number of flips reported by Phylogeny_BnB:', results.best_node.state[-1])
    icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(results.best_node.state[0])
    print('Is the output matrix reported by Phylogeny_BnB conflict free:', icf)

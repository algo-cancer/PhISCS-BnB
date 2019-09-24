import pybnb
import numpy as np
from utils import *
import time
from argparse import ArgumentParser
import networkx as nx
parser = ArgumentParser()
parser.add_argument('-n', '--numberOfCells', dest='n', help='', type=int, default=5)
parser.add_argument('-m', '--numberOfMutations', dest='m', help='', type=int, default=5)
parser.add_argument('-fn', '--falseNegativeRate', dest='fn', help='', type=float, default=0.2)
args = parser.parse_args()


# noisy = np.random.randint(2, size=(args.n, args.m))
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
# noisy = np.array([
#     [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
#     [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
#     [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
#     [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
#     [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#     [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
#     [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
#     [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
#     [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#     [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#     [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
#     [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
#     [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#     [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1]
# ])
# ms_package_path = '/home/frashidi/software/bin/ms'
# ground, noisy, (countFN,countFP,countNA) = get_data(n=args.n, m=args.m,
#                                                     seed=10,fn=args.fn,fp=0,na=0,
#                                                     ms_package_path=ms_package_path)
# print('Number of flips introduced in I: fn={}, fp={}, na={}'.format(countFN, countFP, countNA))
# print(repr(noisy))

solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1), c_time = PhISCS_I(noisy, beta=0.9, alpha=0.00000001)

# csp_solver_path = '/home/frashidi/software/temp/csp_solvers/maxino/code/build/release/maxino'
# a = time.time()
# solution = PhISCS_B(noisy, beta=0.9, alpha=0.00000001, csp_solver_path=csp_solver_path)
# b = time.time()
# print('PhISCS_B in seconds: {:.3f}'.format(b-a))
# print('Number of flips reported by PhISCS_B:', len(np.where(solution != noisy)[0]))

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I):
    def sort_bin(a):
        b = np.transpose(a)
        b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
        idx = np.argsort(b_view.ravel())[::-1]
        c = b[idx]
        return np.transpose(c), idx

    O, idx = sort_bin(I)
    #TODO: delete duplicate columns
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
    # for (u, v, wt) in G.edges.data('weight'):
    #     print(u, v, wt)
    lb = 0
    best_pair_qp, best_pair_w = (None, None), 0
    for a, b in best_pairing:
        # print(a,b,G[a][b]["weight"])
        if G[a][b]["weight"] > best_pair_w:
            best_pair_w = G[a][b]["weight"]
            best_pair_qp = (a, b)
        lb += G[a][b]["weight"]
    return lb, G, best_pair_qp

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def apply_flips(I, F):
    for i,j in F:
        I[i,j] = 1
    return I

def deapply_flips(I, F):
    for i,j in F:
        I[i,j] = 0
    return I

class Phylogeny_BnB(pybnb.Problem):
    def __init__(self, I):
        self.I = I
        self.F = []
        self.nzero = len(np.where(self.I == 0)[0])
        self.lb, self.G, self.best_pair = get_lower_bound(self.I, None, None)
        self.nflip = 0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.lb == self.nflip:
            return self.nflip
        else:
            return self.nzero - self.nflip

    def bound(self):
        return self.lb

    def save_state(self, node):
        node.state = (self.F, self.G, self.best_pair, self.lb, self.nflip)

    def load_state(self, node):
        self.F, self.G, self.best_pair, self.lb, self.nflip = node.state

    def branch(self):
        p, q = self.best_pair
        I = apply_flips(self.I, self.F)
        p,q,oneone,zeroone,onezero = get_a_coflict(I, p, q)
        
        node_l = pybnb.Node()
        G = self.G.copy()
        F = self.F.copy()
        F.append((onezero,q))
        I[onezero,q] = 1
        lb, new_G, new_best_pair = get_lower_bound(I, q, G)
        self.lb = max(self.lb, self.nflip+1+lb)
        node_l.state = (F, new_G, new_best_pair, self.lb, self.nflip+1)
        node_l.queue_priority = -self.lb
        I[onezero,q] = 0
        
        node_r = pybnb.Node()
        G = self.G.copy()
        F = self.F.copy()
        F.append((zeroone,p))
        I[zeroone,p] = 1
        lb, new_G, new_best_pair = get_lower_bound(I, p, G)
        self.lb = max(self.lb, self.nflip+1+lb)
        node_r.state = (F, new_G, new_best_pair, self.lb, self.nflip+1)
        node_r.queue_priority = -self.lb
        
        self.I = deapply_flips(I, F)
        return [node_l, node_r]


problem = Phylogeny_BnB(noisy)
a = time.time()
results = pybnb.solve(problem, log_interval_seconds=10.0, queue_strategy='custom')
b = time.time()
print('PhISCS_I in seconds: {:.3f}'.format(c_time))
print('Number of flips reported by PhISCS_I:', len(np.where(solution != noisy)[0]))
print('Phylogeny_BnB in seconds: {:.3f}'.format(b-a))
if results.solution_status != 'unknown':
    print('Number of flips reported by Phylogeny_BnB:', results.best_node.state[-1])
    I = apply_flips(noisy, results.best_node.state[0])
    icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
    print('Is the output matrix reported by Phylogeny_BnB conflict free:', icf)

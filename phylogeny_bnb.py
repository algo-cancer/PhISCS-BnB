import pybnb
import numpy as np
from utils import *
from phylogeny_lb import *
import time
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-n', '--numberOfCells', dest='n', help='', type=int, default=5)
parser.add_argument('-m', '--numberOfMutations', dest='m', help='', type=int, default=5)
parser.add_argument('-fn', '--falseNegativeRate', dest='fn', help='', type=float, default=0.2)
args = parser.parse_args()


# noisy = np.random.randint(2, size=(args.n, args.m))
# ms_package_path = '/home/frashidi/software/bin/ms'
# ground, noisy, (countFN,countFP,countNA) = get_data(n=args.n, m=args.m,
#                                                     seed=10,fn=args.fn,fp=0,na=0,
#                                                     ms_package_path=ms_package_path)
# print(repr(noisy))

(countFN,countFP,countNA) = (0,0,0)
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
#     [0,1,1,0],
#     [1,0,0,1],
#     [1,1,0,0],
#     [0,0,1,0]
# ])
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

solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1), ci_time = PhISCS_I(noisy, beta=0.9, alpha=0.00000001)
csp_solver_path = '/data/frashidi/_Archived/1_PhISCS/_src/solver/open-wbo/open-wbo_glucose4.1_static'
solution, cb_time = PhISCS_B(noisy, beta=0.9, alpha=0.00000001, csp_solver_path=csp_solver_path)

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

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
    def __init__(self, I, bounding_alg):
        self.I = I
        self.bounding_alg = bounding_alg
        self.F = []
        self.nzero = len(np.where(self.I == 0)[0])
        self.lb, self.G, self.best_pair = self.bounding_alg(self.I, None, None)
        self.nflip = 0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.lb == 0:
            return self.nflip
        else:
            return self.nzero - self.nflip

    def bound(self):
        return self.nflip+self.lb+1

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
        new_lb, new_G, new_best_pair = self.bounding_alg(I, q, G)
        node_l.state = (F, new_G, new_best_pair, new_lb, self.nflip+1)
        node_l.queue_priority = -new_lb
        I[onezero,q] = 0
        
        node_r = pybnb.Node()
        G = self.G.copy()
        F = self.F.copy()
        F.append((zeroone,p))
        I[zeroone,p] = 1
        new_lb, new_G, new_best_pair = self.bounding_alg(I, p, G)
        node_r.state = (F, new_G, new_best_pair, new_lb, self.nflip+1)
        node_r.queue_priority = -new_lb

        self.I = deapply_flips(I, F)
        return [node_l, node_r]


# problem = Phylogeny_BnB(noisy, lb_max_weight_matching)
problem = Phylogeny_BnB(noisy, lb_openwbo)
# problem = Phylogeny_BnB(noisy, lb_gurobi)
# problem = Phylogeny_BnB(noisy, lb_greedy)
# problem = Phylogeny_BnB(noisy, lb_random)

a = time.time()
results = pybnb.solve(problem, log_interval_seconds=10.0, queue_strategy='custom')
b = time.time()
print('Number of flips introduced in I: fn={}, fp={}, na={}'.format(countFN, countFP, countNA))
print('PhISCS_I in seconds: {:.3f}'.format(ci_time))
print('Number of flips reported by PhISCS_I:', len(np.where(solution != noisy)[0]))
print('PhISCS_B in seconds: {:.3f}'.format(cb_time))
print('Number of flips reported by PhISCS_B:', len(np.where(solution != noisy)[0]))
print('Phylogeny_BnB in seconds: {:.3f}'.format(b-a))
if results.solution_status != 'unknown':
    print('Number of flips reported by Phylogeny_BnB:', results.best_node.state[-1])
    I = apply_flips(noisy, results.best_node.state[0])
    icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
    print('Is the output matrix reported by Phylogeny_BnB conflict free:', icf)

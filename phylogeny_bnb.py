from Utils.const import *

import pybnb
import numpy as np
from Utils.util import *
from phylogeny_lb import *
import time
from argparse import ArgumentParser



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
        self.lb, self.G, self.best_pair, self.icf = self.bounding_alg(self.I, None, None)
        self.nflip = 0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.nflip
        else:
            return self.nzero - self.nflip

    def bound(self):
        return self.nflip+self.lb

    def save_state(self, node):
        node.state = (self.F, self.G, self.icf, self.best_pair, self.lb, self.nflip)

    def load_state(self, node):
        self.F, self.G, self.icf, self.best_pair, self.lb, self.nflip = node.state

    def branch(self):
        p, q = self.best_pair
        I = apply_flips(self.I, self.F)
        # icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        # if icf:
        #     return
        p,q,oneone,zeroone,onezero = get_a_coflict(I, p, q)
        
        node_l = pybnb.Node()
        G = self.G.copy()
        F = self.F.copy()
        F.append((onezero,q))
        I[onezero,q] = 1
        new_lb, new_G, new_best_pair, new_icf = self.bounding_alg(I, q, G)
        node_l.state = (F, new_G, new_icf, new_best_pair, new_lb, self.nflip+1)
        node_l.queue_priority = -new_lb
        I[onezero,q] = 0
        
        node_r = pybnb.Node()
        G = self.G.copy()
        F = self.F.copy()
        F.append((zeroone,p))
        I[zeroone,p] = 1
        new_lb, new_G, new_best_pair, new_icf = self.bounding_alg(I, p, G)
        node_r.state = (F, new_G, new_icf, new_best_pair, new_lb, self.nflip+1)
        node_r.queue_priority = -new_lb

        self.I = deapply_flips(I, F)
        return [node_l, node_r]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--numberOfCells', dest='n', help='', type=int, default=5)
    parser.add_argument('-m', '--numberOfMutations', dest='m', help='', type=int, default=5)
    parser.add_argument('-fn', '--falseNegativeRate', dest='fn', help='', type=float, default=0.2)
    args = parser.parse_args()

    noisy = np.random.randint(2, size=(args.n, args.m))
    # ground, noisy, (countFN,countFP,countNA) = get_data(n=args.n, m=args.m, seed=10, fn=args.fn, fp=0, na=0)

    # (countFN,countFP,countNA) = (0,0,0)
    # noisy = np.array([
    #     [0,1,0,0,0,0,1,1,1,0],
    #     [0,1,1,0,1,1,1,0,1,0],
    #     [1,0,0,1,0,1,1,1,0,0],
    #     [1,0,0,0,0,0,0,1,0,0],
    #     [1,1,1,1,1,1,0,1,0,1],
    #     [0,1,1,1,1,1,1,1,0,0],
    #     [1,0,0,1,0,1,0,0,0,0],
    #     [1,1,1,1,0,0,1,0,1,1],
    #     [0,0,1,0,1,1,1,1,1,0],
    #     [1,1,1,1,0,0,1,0,1,1],
    # ])
    # print(repr(noisy))

    solution, (f_0_1_i, f_1_0_i, f_2_0_i, f_2_1_i), ci_time = PhISCS_I(noisy, beta=0.98, alpha=0.00000001)
    solution, (f_0_1_b, f_1_0_b, f_2_0_b, f_2_1_b), cb_time = PhISCS_B(noisy)

    # problem = Phylogeny_BnB(noisy, lb_lp)
    problem = Phylogeny_BnB(noisy, lb_max_weight_matching)
    # problem = Phylogeny_BnB(noisy, lb_phiscs_b)
    # problem = Phylogeny_BnB(noisy, lb_openwbo)
    # problem = Phylogeny_BnB(noisy, lb_gurobi)
    # problem = Phylogeny_BnB(noisy, lb_greedy)
    # problem = Phylogeny_BnB(noisy, lb_random)

    solver = pybnb.Solver()
    a = time.time()
    results = solver.solve(problem,
                            # log=None,
                            log_interval_seconds=10,
                            queue_strategy='custom',
                            # objective_stop=20,
                            # time_limit=0.4
                          )
    b = time.time()
    # queue = solver.save_dispatcher_queue()
    # print(len(queue.nodes))
    # print(results.termination_condition)
    # print('Number of flips introduced in I: fn={}, fp={}, na={}'.format(countFN, countFP, countNA))
    print('PhISCS_I in seconds: {:.3f}'.format(ci_time))
    print('Number of flips reported by PhISCS_I:', f_0_1_i)
    print('PhISCS_B in seconds: {:.3f}'.format(cb_time))
    print('Number of flips reported by‌ PhISCS_B:', f_0_1_b)
    print('Phylogeny_BnB in seconds: {:.3f}'.format(b-a))
    if results.solution_status != 'unknown':
        print('Number of flips reported by Phylogeny_BnB:', results.best_node.state[-1])
        I = apply_flips(noisy, results.best_node.state[0])
        icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        print('Is the output matrix reported by Phylogeny_BnB conflict free:', icf)
        print('Number of nodes processed by Phylogeny_BnB:', results.nodes)

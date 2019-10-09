#!/usr/bin/env python

from argparse import ArgumentParser
import sys
sys.path.append('..')
from phylogeny_lb import *
from Utils.const import *
from Utils.util import *

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
    t1 = time.time()
    for i,j in F:
        I[i,j] = 1
    t2 = time.time()
    return I, t2-t1

def deapply_flips(I, F):
    t1 = time.time()
    for i,j in F:
        I[i,j] = 0
    t2 = time.time()
    return I, t2-t1


class Phylogeny_BnB(pybnb.Problem):
    def __init__(self, I, bounding_alg, bounding_type):
        self.I = I
        self.bounding_alg = bounding_alg
        self.F = []
        self.lb, self.G, self.best_pair, self.icf, self.time1, self.time2, self.time3 = self.bounding_alg(self.I, None, None)
        # print(self.time2)
        self.nflip = 0
        self.bounding_type = bounding_type
        self.time4 = 0.0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.nflip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        return self.nflip+self.lb

    def save_state(self, node):
        node.state = (self.F, self.G, self.icf, self.best_pair, self.lb, self.nflip)

    def load_state(self, node):
        self.F, self.G, self.icf, self.best_pair, self.lb, self.nflip = node.state

    def branch(self):
        p, q = self.best_pair
        I, time4 = apply_flips(self.I, self.F)
        self.time4 += time4
        p,q,oneone,zeroone,onezero = get_a_coflict(I, p, q)
        
        nodes = []
        for r,c in [(onezero,q), (zeroone,p)]:
            node = pybnb.Node()
            F = self.F.copy()
            F.append((r,c))
            I[r,c] = 1
            if self.bounding_type == 'lb_lp_gurobi' or self.bounding_type == 'lb_lp_ortools':
                new_lb, new_G, new_best_pair, new_icf, time1, time2, time3 = self.bounding_alg(I, F, self.G)
                # print(time2)
                # print(new_G.getVarByName('B[{0},{1},1,1]'.format(0, 1)).X)
            else:
                G = self.G.copy()
                new_lb, new_G, new_best_pair, new_icf, time1, time2, time3 = self.bounding_alg(I, c, G)
            self.time1 += time1
            self.time2 += time2
            self.time3 += time3
            node.state = (F, new_G, new_icf, new_best_pair, new_lb, self.nflip+1)
            node.queue_priority = -new_lb
            I[r,c] = 0
            nodes.append(node)
        
        self.I, time4 = deapply_flips(I, F)
        self.time4 += time4
        return nodes


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--numberOfCells', dest='n', help='', type=int, default=5)
    parser.add_argument('-m', '--numberOfMutations', dest='m', help='', type=int, default=5)
    parser.add_argument('-fn', '--falseNegativeRate', dest='fn', help='', type=float, default=0.2)
    args = parser.parse_args()

    # noisy = np.random.randint(2, size=(args.n, args.m))
    ground, noisy, (countFN,countFP,countNA) = get_data(n=args.n, m=args.m, seed=int(100*time.time())%10000, 
                                                        fn=args.fn, fp=0, na=0)

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

    solution, (f_0_1_i, f_1_0_i, f_2_0_i, f_2_1_i), ci_time = PhISCS_I(noisy, beta=0.90, alpha=0.00000001)
    solution, (f_0_1_b, f_1_0_b, f_2_0_b, f_2_1_b), cb_time = PhISCS_B(noisy)

    st = time.time()
    problem = Phylogeny_BnB(noisy, lb_lp_gurobi, 'lb_lp_gurobi')
    # problem = Phylogeny_BnB(noisy, lb_max_weight_matching, 'lb_max_weight_matching')
    # problem = Phylogeny_BnB(noisy, lb_lp_ortools, 'lb_lp_ortools')
    ## TODO: don't use the following bounding yet
    # problem = Phylogeny_BnB(noisy, lb_phiscs_b)
    # problem = Phylogeny_BnB(noisy, lb_openwbo)
    # problem = Phylogeny_BnB(noisy, lb_gurobi)
    # problem = Phylogeny_BnB(noisy, lb_greedy)
    # problem = Phylogeny_BnB(noisy, lb_random)

    solver = pybnb.Solver()
    results = solver.solve(problem,
                            log=None,
                            log_interval_seconds=10,
                            queue_strategy='custom',
                            # objective_stop=20,
                            # time_limit=0.4
                          )
    et = time.time()
    # queue = solver.save_dispatcher_queue()
    # print(len(queue.nodes))
    # print(results.termination_condition)
    # print('Number of flips introduced in I: fn={}, fp={}, na={}'.format(countFN, countFP, countNA))
    # print('TIME Model Preparation in seconds: {:.3f}'.format(problem.time1))
    # print('TIME Model Solvation in seconds: {:.3f}'.format(problem.time2))
    # print('TIME Gusfield in seconds: {:.3f}'.format(problem.time3))
    # print('TIME Preparing I everytime in seconds: {:.3f}'.format(problem.time4))
    # print('PhISCS_I in seconds: {:.3f}'.format(ci_time))
    # print('Phylogeny_BnB in seconds: {:.3f}'.format(et-st))
    # print('Number of nodes processed by Phylogeny_BnB:', results.nodes)
    # print('TIME Remaining: {:.3f}'.format(et-st-(problem.time1+problem.time2+problem.time3+problem.time4)))
    # print('––––––––––––––––')
    I, _ = apply_flips(noisy, results.best_node.state[0])
    icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
    # print('Is the output matrix reported by Phylogeny_BnB conflict free:', icf)
    # print('Number of flips reported by PhISCS_I:', f_0_1_i)
    # print('Number of flips reported by‌ PhISCS_B:', f_0_1_b)
    # print('Number of flips reported by Phylogeny_BnB:', results.best_node.state[-1])
    # print('PhISCS_B in seconds: {:.3f}'.format(cb_time))
    print(f"{args.n},{args.m},{args.fn:.1f},{countFN},{countFP},{countNA},{f_0_1_i},{f_0_1_b},{results.best_node.state[-1]},{problem.time1:.3f},{problem.time2:.3f},{problem.time3:.3f},{problem.time4:.3f},{et-st-(problem.time1+problem.time2+problem.time3+problem.time4):.3f},{et-st:.3f},{ci_time:.3f},{cb_time:.3f},{results.nodes},{icf}")

import numpy as np
from utils import *
import pybnb
import operator
from collections import defaultdict
from itertools import chain, combinations

ms_package_path = '/home/frashidi/software/bin/ms'

ground, noisy, (countFN,countFP,countNA) = get_data(n=10, m=8, seed=1, fn=0.20, fp=0, na=0, 
                                                    ms_package_path=ms_package_path)
print(noisy)
solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1) = PhISCS_I(noisy, beta=0.20, alpha=0.00000001)
# print(solution)
print(np.where(solution != noisy))


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def number_of_conflicts(D):
        noc = 0
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                oneone = 0
                zeroone = 0
                onezero = 0
                for r in range(D.shape[0]):
                    if D[r][p] == 1 and D[r][q] == 1:
                        oneone += 1
                    if D[r][p] == 0 and D[r][q] == 1:
                        zeroone += 1
                    if D[r][p] == 1 and D[r][q] == 0:
                        onezero += 1
                noc += oneone*zeroone*onezero
        return noc

def give_me_the_lower_bound(noisy):
    def important_columns_in_conflicts(D):
        important_columns = defaultdict(lambda: 0)
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                oneone = 0
                zeroone = 0
                onezero = 0
                for r in range(D.shape[0]):
                    if D[r][p] == 1 and D[r][q] == 1:
                        oneone += 1
                    if D[r][p] == 0 and D[r][q] == 1:
                        zeroone += 1
                    if D[r][p] == 1 and D[r][q] == 0:
                        onezero += 1
                if oneone*zeroone*onezero > 0:
                    important_columns[(p,q)] += oneone*zeroone*onezero
        return important_columns
    
    def get_partinion(D):
        icic = important_columns_in_conflicts(D)
        sorted_icic = sorted(icic.items(), key=operator.itemgetter(1), reverse=True)
        pairs = [sorted_icic[0][0]]
        elements = [sorted_icic[0][0][0], sorted_icic[0][0][1]]
        sorted_icic.remove(sorted_icic[0])
        for x in sorted_icic[:]:
            notFound = True
            for y in x[0]:
                if y in elements:
                    sorted_icic.remove(x)
                    notFound = False
                    break
            if notFound:
                pairs.append(x[0])
                elements.append(x[0][0])
                elements.append(x[0][1])
        #print(sorted_icic, pairs, elements)
        partitions = []
        for x in pairs:
            partitions.append(D[:,x])
        return partitions
    
    def give_me_the_lower_bound_helper(D):        
        def conflicts_set(D):
            all_conf = []
            for p in range(D.shape[1]):
                for q in range(p + 1, D.shape[1]):
                    conf_oneone = []
                    conf_zeroone = []
                    conf_onezero = []
                    for r in range(D.shape[0]):
                        if D[r][p] == 1 and D[r][q] == 1:
                            conf_oneone.append(r)
                        if D[r][p] == 0 and D[r][q] == 1:
                            conf_zeroone.append(r)
                        if D[r][p] == 1 and D[r][q] == 0:
                            conf_onezero.append(r)
                    for r1 in conf_oneone:
                        for r2 in conf_zeroone:
                            for r3 in conf_onezero:
                                #print(p,q, r1, r2, r3)
                                all_conf.append(set([r1,r2,r3]))
            return all_conf
        
        def powerset(iterable):
            xs = list(iterable)
            return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))
    
        rows_set = range(D.shape[0])
        for subset in map(set, powerset(set(rows_set))):
            if len(subset) == 0:
                continue
            all_conf = conflicts_set(D)
            if len(all_conf) == 0:
                return 0
            catch_subset = 0
            for conf in all_conf:
                if subset.issubset(conf):
                    catch_subset += 1
            if catch_subset == len(all_conf):
                if len(subset) == 1:
                    return 1
                else:
                    return int(np.ceil(len(subset)/np.log2(len(subset))))
            return int(np.ceil(len(rows_set)/np.log2(len(rows_set))))
    
    #return give_me_the_lower_bound_helper(noisy)
    LB = []
    for D in get_partinion(noisy):
        LB.append(give_me_the_lower_bound_helper(D))
    return sum(LB)


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class PhISCS(pybnb.Problem):
    def __init__(self, I):
        self.I = I
        self.X = np.where(self.I == 0)
        self.flip = 0
        self.idx = 0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        return number_of_conflicts(self.I)

    def bound(self):
        return give_me_the_lower_bound(self.I)

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
            node.state = (I, self.idx+1, self.flip+1)
            yield node
            
            node = pybnb.Node()
            I = self.I.copy()
            x = self.X[0][self.idx]
            y = self.X[1][self.idx]
            I[x, y] = 0
            node.state = (I, self.idx+1, self.flip)
            yield node

problem = PhISCS(noisy)
results = pybnb.solve(problem, relative_gap=0, absolute_gap=0)

print(results.best_node.state)

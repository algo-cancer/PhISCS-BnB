import pybnb
import numpy as np
from utils import *
import operator
import datetime
from collections import defaultdict
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-n', '--numberOfCells', dest='n', help='', type=int, default=5)
parser.add_option('-m', '--numberOfMutations', dest='m', help='', type=int, default=5)
parser.add_option('-r', '--partitionRandomly', dest='r', help='', action='store_true', default=False)
parser.add_option('-w', '--whichAlgorithm', dest='w', help='', type=str, default='b')
options, args = parser.parse_args()


noisy = np.random.randint(2, size=(options.n, options.m))
# print(noisy)
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
# noisy = np.array([
#     [0,0,1,0],
#     [1,0,1,1],
#     [1,1,1,1],
#     [0,1,0,1]
# ])
# noisy = np.array([
#     [0,1,1,0],
#     [1,0,0,1],
#     [1,1,0,0],
#     [0,0,1,0]
# ])
# noisy = np.zeros((4,4))
# ms_package_path = '/home/frashidi/software/bin/ms'
# ground, noisy, (countFN,countFP,countNA) = get_data(n=30, m=15, seed=1, fn=0.20, fp=0, na=0, ms_package_path=ms_package_path)
a = datetime.datetime.now()
solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1) = PhISCS_I(noisy, beta=0.9, alpha=0.00000001)
b = datetime.datetime.now()
c = b - a
print('PhISCS_I in microseconds: ', c.microseconds)
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

def get_lower_bound(noisy, partition_randomly=False):
    def get_important_pair_of_columns_in_conflict(D):
        important_columns = defaultdict(lambda: 0)
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                oneone = 0
                zeroone = 0
                onezero = 0
                for r in range(D.shape[0]):
                    if D[r,p] == 1 and D[r,q] == 1:
                        oneone += 1
                    if D[r,p] == 0 and D[r,q] == 1:
                        zeroone += 1
                    if D[r,p] == 1 and D[r,q] == 0:
                        onezero += 1
                if oneone*zeroone*onezero > 0:
                    important_columns[(p,q)] += oneone*zeroone*onezero
        return important_columns
    
    def get_partition_sophisticated(D):
        ipofic = get_important_pair_of_columns_in_conflict(D)
        if len(ipofic) == 0:
            return []
        sorted_ipofic = sorted(ipofic.items(), key=operator.itemgetter(1), reverse=True)
        pairs = [sorted_ipofic[0][0]]
        elements = [sorted_ipofic[0][0][0], sorted_ipofic[0][0][1]]
        sorted_ipofic.remove(sorted_ipofic[0])
        for x in sorted_ipofic[:]:
            notFound = True
            for y in x[0]:
                if y in elements:
                    sorted_ipofic.remove(x)
                    notFound = False
                    break
            if notFound:
                pairs.append(x[0])
                elements.append(x[0][0])
                elements.append(x[0][1])
        #print(sorted_ipofic, pairs, elements)
        partitions = []
        for x in pairs:
            partitions.append(D[:,x])
        return partitions
    
    def get_partition_random(D):
        d = int(D.shape[1]/2)
        partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
        partitions = []
        for x in partitions_id:
            partitions.append(D[:,x])
        return partitions
    
    def get_lower_bound_for_a_pair_of_columns(D):
        foundOneOne = False
        numberOfZeroOne = 0
        numberOfOneZero = 0
        for r in range(D.shape[0]):
            if D[r,0] == 1 and D[r,1] == 1:
                foundOneOne = True
            if D[r,0] == 0 and D[r,1] == 1:
                numberOfZeroOne += 1
            if D[r,0] == 1 and D[r,1] == 0:
                numberOfOneZero += 1
        if foundOneOne:
            if numberOfZeroOne*numberOfOneZero > 0:
                return min(numberOfZeroOne, numberOfOneZero)
        return 0
    
    LB = []
    if partition_randomly:
        partitions = get_partition_random(noisy)
    else:
        partitions = get_partition_sophisticated(noisy)
    for D in partitions:
        LB.append(get_lower_bound_for_a_pair_of_columns(D))
    return sum(LB)

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class Phylogeny_BnB_a(pybnb.Problem):
    def __init__(self, I):
        self.I = I
        self.X = np.where(self.I == 0)
        self.flip = 0
        self.idx = 0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        if icf:
            return self.flip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        return self.flip + get_lower_bound(self.I, partition_randomly=options.r)

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


class Phylogeny_BnB_b(pybnb.Problem):
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
        return self.nflip + get_lower_bound(self.I, partition_randomly=options.r)

    def save_state(self, node):
        node.state = (self.I, self.nflip)

    def load_state(self, node):
        self.I, self.nflip = node.state

    def branch(self):
        icf, (p,q) = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        p,q,oneone,zeroone,onezero = get_a_coflict(self.I, p, q)
        
        node = pybnb.Node()
        I = self.I.copy()
        I[onezero,q] = 1
        node.state = (I, self.nflip+1)
        yield node
        
        node = pybnb.Node()
        I = self.I.copy()
        I[zeroone,p] = 1
        node.state = (I, self.nflip+1)
        yield node


class Phylogeny_BnB_c(pybnb.Problem):
    def __init__(self, I):
        self.I = I
        self.nflip = 0
        self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        self.boundVal = 0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.nflip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        newBound = self.nflip + get_lower_bound(self.I, partition_randomly=options.r)
        self.boundVal = max(self.boundVal, newBound)
        return self.boundVal

    def save_state(self, node):
        node.state = (self.I, self.icf, self.colPair, self.boundVal, self.nflip)

    def load_state(self, node):
        self.I, self.icf, self.colPair, self.boundVal, self.nflip = node.state

    def branch(self):
        if self.icf:
            return
        p, q = self.colPair
        p,q,oneone,zeroone,onezero = get_a_coflict(self.I, p, q)
        
        node = pybnb.Node()
        I = self.I.copy()
        I[onezero,q] = 1
        newIcf, newColPar = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        node.state = (I, newIcf, newColPar, self.boundVal, self.nflip+1)
        yield node
        
        node = pybnb.Node()
        I = self.I.copy()
        I[zeroone,p] = 1
        newIcf, newColPar = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
        node.state = (I, newIcf, newColPar, self.boundVal, self.nflip+1)
        yield node


class Phylogeny_BnB_d(pybnb.Problem):
    def __init__(self, I):
        self.I = I
        self.nflip = 0
        self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        self.boundVal = 0
    
    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.nflip
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        newBound = self.nflip + get_lower_bound(self.I, partition_randomly=options.r)
        self.boundVal = max(self.boundVal, newBound)
        return self.boundVal

    def save_state(self, node):
        node.state = (self.icf, self.colPair, self.boundVal, self.nflip)

    def load_state(self, node):
        self.icf, self.colPair, self.boundVal, self.nflip = node.state

    def branch(self):
        if self.icf:
            return
        p, q = self.colPair
        p,q,oneone,zeroone,onezero = get_a_coflict(self.I, p, q)
        
        node = pybnb.Node()
        self.I[onezero,q] = 1
        newIcf, newColPar = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        node.state = (newIcf, newColPar, self.boundVal, self.nflip+1)
        yield node
        self.I[onezero,q] = 0
        
        node = pybnb.Node()
        self.I[zeroone,p] = 1
        newIcf, newColPar = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        node.state = (newIcf, newColPar, self.boundVal, self.nflip+1)
        yield node
        self.I[zeroone,p] = 0



if options.w == 'a':
    print('Phylogeny_BnB_a is chosen')
    problem = Phylogeny_BnB_a(noisy)
elif options.w == 'b':
    print('Phylogeny_BnB_b is chosen')
    problem = Phylogeny_BnB_b(noisy)
elif options.w == 'c':
    print('Phylogeny_BnB_c is chosen')
    problem = Phylogeny_BnB_c(noisy)
elif options.w == 'd':
    print('Phylogeny_BnB_d is chosen')
    problem = Phylogeny_BnB_d(noisy)
else:
    print('Wrong Algorithm')

a = datetime.datetime.now()
results = pybnb.solve(problem, log_interval_seconds=10.0)
b = datetime.datetime.now()
c = b - a
print('Phylogeny_BnB in microseconds:', c.microseconds)
# print(results.solution_status, type(results.solution_status))
if results.solution_status != "unknown":
    print('Number of flips reported by Phylogeny_BnB:', results.best_node.state[-1])
    icf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(results.best_node.state[0])
    print('Is the output matrix reported by Phylogeny_BnB conflict free:', icf)

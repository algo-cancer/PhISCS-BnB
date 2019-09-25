import numpy as np
from utils import *
import networkx as nx
import subprocess
from collections import defaultdict

csp_solver_path = './openwbo'

def lb_csp(D, changed_column, previous_G):
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

    clause_soft = defaultdict('')
    for (u, v, wt) in G.edges.data('weight'):
        if u < v:
            clause_soft[(u,v)]

    outfile = 'cnf.tmp'
    with open(outfile, 'w') as out:
        out.write('p wcnf {} {} {}\n'.format(numVarY+numVarX+numVarB, len(clauseSoft)+len(clauseHard), hardWeight))
        
        for i in range(D.shape[1]):
            cnf = ''
            for j in range(i, D.shape[1]):
                numVarX = 0
                cnf += '{}'.format(numVarX)
                out.write('{} 0\n'.format(cnf))
                G[i][j]
    
    command = '{} {}'.format(csp_solver_path, outfile)
    proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = proc.communicate()

    variables = output.decode().split('\n')[-2][2:].split(' ')
    print(variables)
    
    print(best_pairing)
    best_pair_qp, best_pair_w = (None, None), 0
    lb = 0
    return lb, G, best_pair_qp

# print(lb_csp(np.zeros((3,2)), None, None))

def lb_greedy(D, a, b):
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
                if oneone > 0:
                    important_columns[(p,q)] += min(zeroone, onezero)
        return important_columns
    
    ipofic = get_important_pair_of_columns_in_conflict(D)
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
    lb = 0
    for x in pairs:
        lb += ipofic[x]
    
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return lb, {}, best_pair_qp
    

def lb_random(D, a, b):
    def get_partition_random(D):
        d = int(D.shape[1]/2)
        partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
        return partitions_id
    
    def calc_min0110_for_one_pair_of_columns(D, p, q):
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
            return min(numberOfZeroOne, numberOfOneZero)
        else:
            return 0

    lb = 0
    for x in get_partition_random(D):
        lb += calc_min0110_for_one_pair_of_columns(D, x[0], x[1])
    
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return lb, {}, best_pair_qp


def lb_phiscs_b(D, a, b):
    # def get_partition_random(D, n_group_members=5):
    #     d = int(D.shape[1]/n_group_members)
    #     partitions_id = np.random.choice(range(D.shape[1]), size=(d, n_group_members), replace=False)
    #     return partitions_id
    def blockshaped(arr, nrows, ncols):
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))
    
    lb = 0
    for block in blockshaped(D, D.shape[0], 5):
        solution, c_time = PhISCS_B(block, beta=0.9, alpha=0.00000001, csp_solver_path=csp_solver_path)
        lb += len(np.where(solution != block)[0])
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return lb, {}, best_pair_qp


def lb_openwbo(D, a, b):
    solution, c_time = PhISCS_B(D, beta=0.9, alpha=0.00000001, csp_solver_path=csp_solver_path)
    lb = len(np.where(solution != D)[0])
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return lb, {}, best_pair_qp


def lb_gurobi(D, a, b):
    solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1), c_time = PhISCS_I(D, beta=0.9, alpha=0.00000001)
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return flips_0_1, {}, best_pair_qp


def lb_max_weight_matching(D, changed_column, previous_G):
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
    best_pair_qp, best_pair_w = (None, None), 0
    # for (u, v, wt) in G.edges.data('weight'):
    #     if wt > best_pair_w:
    #         best_pair_qp = (u, v)
    #         best_pair_w = wt
        # print(u, v, wt)
    lb = 0
    # best_pair_qp, best_pair_w = (None, None), 0
    for a, b in best_pairing:
        # print(a,b,G[a][b]["weight"])
        if G[a][b]["weight"] > best_pair_w:
            best_pair_w = G[a][b]["weight"]
            best_pair_qp = (a, b)
        lb += G[a][b]["weight"]
    return lb, G, best_pair_qp
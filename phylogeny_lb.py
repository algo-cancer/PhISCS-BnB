import numpy as np
from utils import *
import networkx as nx

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
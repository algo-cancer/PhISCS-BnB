from funcs import is_conflict_free_gusfield_and_get_two_columns_in_coflicts
import numpy as np
from funcs import get_lower_bound
import networkx as nx
from collections import defaultdict
import operator
import time
import pandas as pd
from tqdm import tqdm
def get_lower_bound(noisy, partition_randomly=False):
  def get_important_pair_of_columns_in_conflict(D):
    important_columns = defaultdict(lambda: 0)
    for p in range(D.shape[1]):
      for q in range(p + 1, D.shape[1]):
        oneone = 0
        zeroone = 0
        onezero = 0
        for r in range(D.shape[0]):
          if D[r, p] == 1 and D[r, q] == 1:
            oneone += 1
          if D[r, p] == 0 and D[r, q] == 1:
            zeroone += 1
          if D[r, p] == 1 and D[r, q] == 0:
            onezero += 1
        if oneone * zeroone * onezero > 0:
          important_columns[(p, q)] += oneone * zeroone * onezero
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
    # print(sorted_ipofic, pairs, elements)
    partitions = []
    for x in pairs:
      partitions.append(D[:, x])
    return partitions

  def get_partition_random(D):
    d = int(D.shape[1] / 2)
    partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
    partitions = []
    for x in partitions_id:
      partitions.append(D[:, x])
    return partitions

  def get_lower_bound_for_a_pair_of_columns(D):
    foundOneOne = False
    numberOfZeroOne = 0
    numberOfOneZero = 0
    for r in range(D.shape[0]):
      if D[r, 0] == 1 and D[r, 1] == 1:
        foundOneOne = True
      if D[r, 0] == 0 and D[r, 1] == 1:
        numberOfZeroOne += 1
      if D[r, 0] == 1 and D[r, 1] == 0:
        numberOfOneZero += 1
    if foundOneOne:
      if numberOfZeroOne * numberOfOneZero > 0:
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

def get_lower_bound_for_a_pair_of_columns(D):
  foundOneOne = False
  numberOfZeroOne = 0
  numberOfOneZero = 0
  for r in range(D.shape[0]):
    if D[r, 0] == 1 and D[r, 1] == 1:
      foundOneOne = True
    if D[r, 0] == 0 and D[r, 1] == 1:
      numberOfZeroOne += 1
    if D[r, 0] == 1 and D[r, 1] == 0:
      numberOfOneZero += 1
  if foundOneOne:
    if numberOfZeroOne * numberOfOneZero > 0:
      return min(numberOfZeroOne, numberOfOneZero)
  return 0

def make_graph_weighted(x):
  # print(x)
  G = nx.Graph()
  for i in range(x.shape[1]):
    for j in range(i):
      G.add_edge(i,j, weight = get_lower_bound_for_a_pair_of_columns(x[:, [i, j]]))
  return G

def make_graph(x):
  # print(x)
  G = nx.Graph()
  for i in range(x.shape[1]):
    for j in range(i):
      if get_lower_bound_for_a_pair_of_columns(x[:, [i, j]])>0:
        G.add_edge(i,j)
  return G


def gmLBw(x):
  G = make_graph_weighted(x)
  mat = nx.max_weight_matching(G)
  ret = 0
  for a, b in mat:
    ret += G[a][b]["weight"]
  return ret


def gmLB(x):
  G = make_graph(x)
  mat = nx.maximal_matching(G)
  ret = 0
  for a, b in mat:
    ret += get_lower_bound_for_a_pair_of_columns(x[:, [a, b]])
  return ret

def gmLB3(x):
  ev = []
  G = nx.Graph()
  for i in range(x.shape[1]):
    for j in range(i):
      w = get_lower_bound_for_a_pair_of_columns(x[:, [i, j]])
      G.add_edge(i, j, weight=w)
      ev.append(w)
  t = np.choice(ev, 1)
  G1 = nx.Graph()


if __name__ == '__main__':
  n = 10
  for m in range(10, 200, 10):
    for i in range(5):
      Ir = np.random.randint(2, size=(n, m))
      # print(Ir)
      # G = make_graph(Ir)
      # print(G.edges(data = True))
      # mat = nx.max_weight_matching(G)
      # print(mat)

      # print(nx.adjacency_matrix(G))
      t = []
      t.append(time.time())
      xt = get_lower_bound(Ir, True)
      t.append(time.time())
      xf = get_lower_bound(Ir, False)
      t.append(time.time())
      a, b = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(Ir)
      t.append(time.time())
      g = gmLB(Ir)
      t.append(time.time())
      gw = gmLBw(Ir)
      t.append(time.time())
      print(xt, xf, a, g, gw, sep= ",")
      # print(m, t[1]-t[0], t[2]-t[1], t[3]-t[2], t[4]-t[3], sep= ",")

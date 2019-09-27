import sys
sys.path.append('..')
import os
import subprocess
import numpy as np
import random
import math
from gurobipy import *
import time
from lp_bounding import LP_brief
import networkx as nx
from interfaces import *
import copy
import scipy.sparse as sp

class DynamicMWMBounding(BoundingAlgAbstract):
  def __init__(self):
    self.matrix = None
    self.G = None

  def reset(self, matrix):
    self.matrix = matrix
    self.G = nx.Graph()
    for p in range(self.matrix.shape[1]):
      for q in range(p + 1, self.matrix.shape[1]):
        self.calc_min0110_for_one_pair_of_columns(p, q, self.matrix)

  def calc_min0110_for_one_pair_of_columns(self, p, q, currentMatrix):
    foundOneOne = False
    numberOfZeroOne = 0
    numberOfOneZero = 0
    for r in range(currentMatrix.shape[0]):
      if currentMatrix[r, p] == 1 and currentMatrix[r, q] == 1:
        foundOneOne = True
      if currentMatrix[r, p] == 0 and currentMatrix[r, q] == 1:
        numberOfZeroOne += 1
      if currentMatrix[r, p] == 1 and currentMatrix[r, q] == 0:
        numberOfOneZero += 1
    if self.G.has_edge(p, q):
      self.G.remove_edge(p, q)
    if foundOneOne:
      self.G.add_edge(p, q, weight=min(numberOfZeroOne, numberOfOneZero))

  def getBound(self, delta):
    currentMatrix = self.matrix + delta
    oldG = copy.deepcopy(self.G)
    flipsMat = np.transpose(delta.nonzero())
    flippedColsSet = set(flipsMat[:, 1])
    for q in flippedColsSet: # q is a changed column
      for p in range(self.matrix.shape[1]):
        if p < q:
          self.calc_min0110_for_one_pair_of_columns(p, q, currentMatrix)
        elif q < p:
          self.calc_min0110_for_one_pair_of_columns(q, p, currentMatrix)

    best_pairing = nx.max_weight_matching(self.G)
    lb = 0
    for a, b in best_pairing:
      lb += self.G[a][b]["weight"]

    self.G = oldG
    return lb + flipsMat.shape[0]


class StaticMWMBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None):
    self.ratio = ratio
    self.matrix = None


  def reset(self, matrix):
    self.matrix = matrix


  def getBound(self, delta):
    nFlips = delta.count_nonzero()
    currentMatrix = self.matrix + delta
    self.G = nx.Graph()
    for p in range(currentMatrix.shape[1]):
      for q in range(p + 1, currentMatrix.shape[1]):
        self.calc_min0110_for_one_pair_of_columns(p, q, currentMatrix)
    best_pairing = nx.max_weight_matching(self.G)
    lb = 0
    for a, b in best_pairing:
      lb += self.G[a][b]["weight"]

    if self.ratio is None:
      returnValue = lb + nFlips
    else:
      returnValue =  np.int(np.ceil(self.ratio * lb)) + nFlips
    return returnValue

  def calc_min0110_for_one_pair_of_columns(self, p, q, currentMatrix):
    foundOneOne = False
    numberOfZeroOne = 0
    numberOfOneZero = 0
    for r in range(currentMatrix.shape[0]):
      if currentMatrix[r, p] == 1 and currentMatrix[r, q] == 1:
        foundOneOne = True
      if currentMatrix[r, p] == 0 and currentMatrix[r, q] == 1:
        numberOfZeroOne += 1
      if currentMatrix[r, p] == 1 and currentMatrix[r, q] == 0:
        numberOfOneZero += 1
    if self.G.has_edge(p, q):
      self.G.remove_edge(p, q)
    if foundOneOne:
      self.G.add_edge(p, q, weight=min(numberOfZeroOne, numberOfOneZero))


if __name__ == '__main__':
  n, m = 10, 10
  x = np.random.randint(2, size=(n, m))
  delta = sp.lil_matrix((n, m))

  # x = np.array([[0, 1, 1, 1, 1, 1],
  #        [1, 0, 1, 1, 0, 1],
  #        [1, 0, 1, 1, 0, 1],
  #        [0, 0, 0, 0, 1, 1],
  #        [0, 0, 0, 0, 1, 1],
  #        [1, 0, 1, 1, 0, 0],
  #        [0, 0, 0, 0, 1, 0],
  #        [1, 1, 1, 0, 0, 1]])
  # delta = sp.lil_matrix([[0, 0, 0, 0, 0, 0],
  #         [0, 1, 0, 0, 0, 0],
  #         [0, 1, 0, 0, 0, 0],
  #         [0, 1, 1, 1, 0, 0],
  #         [0, 1, 1, 1, 0, 0],
  #         [0, 1, 0, 0, 0, 1],
  #         [0, 0, 0, 1, 0, 0],
  #         [0, 0, 0, 0, 0, 0]], dtype=np.int8)


  ss = StaticMWMBounding()
  ss.reset(x)

  algo = DynamicMWMBounding()
  resetTime = time.time()
  algo.reset(x)
  resetTime = time.time() - resetTime
  print(resetTime)

  print(delta.count_nonzero())
  # print(x+delta)
  print(LP_brief(x+delta), algo.getBound(delta), ss.getBound(delta))


  for t in range(10):
    ind = np.nonzero(1 - (x+delta))
    a, b = ind[0][0], ind[1][0]
    print(a,b)
    delta[a, b] = 1

    # algo.reset(x)
    calcTime = time.time()
    bndAdapt = algo.getBound(delta)
    calcTime = time.time() - calcTime
    print(calcTime)

    ssBnd = ss.getBound(delta)
    print( bndAdapt == ssBnd, bndAdapt, ssBnd)

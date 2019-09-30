import sys
sys.path.append('..') # to get files outside this directory
import numpy as np
import random
import math
from gurobipy import *
import time
from interfaces import *
import scipy.sparse as sp
from funcs import myPhISCS_I


class DynamicLPBounding(BoundingAlgAbstract):
  def __init__(self, ratio=None):
    raise NotImplementedError("The method not implemented")



class SemiDynamicLPBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None, continuous = True):
    self.ratio = ratio
    self.matrix = None
    self.model = None
    self.yVars = None
    self.continuous = continuous

  def getName(self):
    return type(self).__name__+f"_{self.ratio}_{self.continuous}"

  def reset(self, matrix):
    self.matrix = matrix
    self.model, self.yVars = StaticLPBounding.makeGurobiModel(self.matrix, continuous=self.continuous)
    self.model.optimize()


  def _flip(self, c, m):
    self.model.addConstr(self.yVars[c, m] == 1)

  def _unFlipLast(self):
    self.model.remove(self.model.getConstrs()[-1])

  def getBound(self, delta):
    flips = np.transpose(delta.nonzero())
    newConstrs = (self.yVars[flips[i, 0], flips[i, 1]] == 1 for i in range(flips.shape[0]))
    newConstrsReturned = self.model.addConstrs(newConstrs)
    self.model.update()
    self.model.optimize()
    if self.ratio is not None:
      bound = np.int(np.ceil(self.ratio * self.model.objVal))
    else:
      bound = np.int(np.ceil(self.model.objVal))
    for cnstr in newConstrsReturned.values():
      self.model.remove(cnstr)
    self.model.update()
    return bound


class StaticLPBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None, continuous = True):
    self.ratio = ratio
    self.matrix = None
    self.continuous = continuous


  def getName(self):
    return type(self).__name__+f"_{self.ratio}_{self.continuous}"


  def reset(self, matrix):
    self.matrix = matrix

  def getBound(self, delta):
    bound = StaticLPBounding.LP_brief(self.matrix + delta, self.continuous)
    if self.ratio is not None:
      bound = np.int(np.ceil(self.ratio * bound))
    else:
      bound = np.int(np.ceil(bound))

    return bound + delta.count_nonzero()

  @staticmethod
  def LP_brief(I, continuous= True):
    model, Y = StaticLPBounding.makeGurobiModel(I, continuous = continuous)
    return StaticLPBounding.LP_Bounding_From_Model(model)

  @staticmethod
  def LP_Bounding_From_Model(model):
    model.optimize()
    return np.int(np.ceil(model.objVal))


  @staticmethod
  def makeGurobiModel(I, continuous = True):
    if continuous:
      varType = GRB.CONTINUOUS
    else:
      varType = GRB.BINARY

    numCells, numMutations = I.shape

    model = Model(f'LP_{time.time()}')
    model.Params.OutputFlag = 0
    Y = {}
    for c in range(numCells):
      for m in range(numMutations):
        if I[c, m] == 0:
          Y[c, m] = model.addVar(0, 1, obj=1, vtype=varType, name='Y({0},{1})'.format(c, m))
        elif I[c, m] == 1:
          Y[c, m] = 1

    B = {}
    for p in range(numMutations + 1):
      for q in range(numMutations + 1):
        B[p, q, 1, 1] = model.addVar(0, 1, vtype=varType, obj=0,
                                     name='B[{0},{1},1,1]'.format(p, q))
        B[p, q, 1, 0] = model.addVar(0, 1, vtype=varType, obj=0,
                                     name='B[{0},{1},1,0]'.format(p, q))
        B[p, q, 0, 1] = model.addVar(0, 1, vtype=varType, obj=0,
                                     name='B[{0},{1},0,1]'.format(p, q))
    model.update()

    for i in range(numCells):
      for p in range(numMutations):
        for q in range(numMutations):
          model.addConstr(Y[i, p] + Y[i, q] - B[p, q, 1, 1] <= 1)
          model.addConstr(-Y[i, p] + Y[i, q] - B[p, q, 0, 1] <= 0)
          model.addConstr(Y[i, p] - Y[i, q] - B[p, q, 1, 0] <= 0)
    for p in range(numMutations + 1):
      model.addConstr(B[p, numMutations, 1, 0] == 0)
    for p in range(numMutations):
      for q in range(numMutations):
        model.addConstr(B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2)

    model.Params.ModelSense = GRB.MINIMIZE
    model.update()
    return model, Y


class StaticILPBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None):
    self.ratio = ratio
    self.matrix = None


  def reset(self, matrix):
    self.matrix = matrix

  def getBound(self, delta):
    model, Y = StaticLPBounding.makeGurobiModel(self.matrix + delta, continuous= False)
    optim = StaticLPBounding.LP_Bounding_From_Model(model)
    if self.ratio is not None:
      bound = np.int(np.ceil(self.ratio * optim))
    else:
      bound = np.int(np.ceil(optim))
    return bound + delta.count_nonzero()


if __name__ == '__main__':

  # n, m = 10, 10
  # x = np.random.randint(2, size=(n, m))
  # delta = sp.lil_matrix((n, m ))

  # x = np.array([[1, 0, 1, 0, 0],
  #        [1, 1, 1, 1, 0],
  #        [1, 1, 0, 1, 0],
  #        [0, 1, 0, 1, 1],
  #        [0, 0, 1, 1, 1]], dtype=np.int8)
  #
  # delta = sp.lil_matrix([[0, 0, 0, 0, 0],
  #         [0, 0, 0, 0, 0],
  #         [0, 0, 1, 0, 0],
  #         [1, 0, 0, 0, 0],
  #         [0, 0, 0, 0, 0]], dtype=np.int8)

  # delta = np.array([[0, 0, 0, 0, 0],
  #       [0, 0, 0, 0, 0],
  #       [0, 0, 1, 0, 0],
  #       [1, 0, 1, 0, 0],
  #       [0, 0, 0, 0, 0]], dtype=np.int8)
  ss = StaticLPBounding()
  ss.reset(x)

  algo = SemiDynamicLPBounding()
  resetTime = time.time()
  algo.reset(x)
  resetTime = time.time() - resetTime
  print(resetTime)

  xp = np.asarray(x + delta)
  print(type(xp))
  optim = myPhISCS_I(xp)
  print("Optimal answer:", optim)
  print(len(algo.model.getConstrs()))
  print(StaticLPBounding.LP_brief(xp), algo.getBound(delta), ss.getBound(delta))
  print(len(algo.model.getConstrs()))

  for _ in range(5):
    print(ss.getBound(delta))
  exit(0)
  for t in range(3):
    ind = np.nonzero(1 - (x+delta))
    a, b = ind[0][0], ind[1][0]
    delta[a, b] = 1

    calcTime = time.time()
    bndAdapt = algo.getBound(delta)
    calcTime = time.time() - calcTime
    print(calcTime)
    bndFull = StaticLPBounding.LP_brief(x+delta) + t + 1
    print(bndFull == bndAdapt, bndFull, bndAdapt)
    ssBnd = ss.getBound(delta)
    print(bndFull == ssBnd, ssBnd)

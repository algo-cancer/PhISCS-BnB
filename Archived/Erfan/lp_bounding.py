import os
import subprocess
import numpy as np
import random
import math
from gurobipy import *
import time

def calcNF(obj, n0, n1, alpha = 0.0001, beta = 0.99):
    nom = obj - (n1 * np.log(1 - beta) + n0 * np.log(1 - alpha))
    den = np.log(beta / (1 - alpha))
    return nom / den


def makeGurobiModel(I):
  numCells, numMutations = I.shape

  model = Model(f'LP_{time.time()}')
  model.Params.OutputFlag = 0
  Y = {}
  for c in range(numCells):
    for m in range(numMutations):
      if I[c, m] == 0:
        Y[c, m] = model.addVar(0, 1, obj=1, vtype=GRB.CONTINUOUS, name='Y({0},{1})'.format(c, m))
      elif I[c, m] == 1:
        Y[c, m] = 1

  B = {}
  for p in range(numMutations + 1):
    for q in range(numMutations + 1):
      B[p, q, 1, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
                                   name='B[{0},{1},1,1]'.format(p, q))
      B[p, q, 1, 0] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
                                   name='B[{0},{1},1,0]'.format(p, q))
      B[p, q, 0, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
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
  return model, Y

def flip(c, m, model, Y):
  model.addConstr(Y[c, m] == 1)


def unFlipLast(model):
  model.remove(model.getConstrs()[-1])


def LP_Bounding(I):
  beta, alpha = 0.99, 0.0001
  numCells, numMutations = I.shape
  sol_Y = []
  sol_K = []

  model = Model('LP_2')
  model.Params.OutputFlag = 0
  Y = {}
  for c in range(numCells):
    for m in range(numMutations):
      Y[c, m] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, name='Y({0},{1})'.format(c, m))
  B = {}
  for p in range(numMutations + 1):
    for q in range(numMutations + 1):
      B[p, q, 1, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
                                   name='B[{0},{1},1,1]'.format(p, q))
      B[p, q, 1, 0] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
                                   name='B[{0},{1},1,0]'.format(p, q))
      B[p, q, 0, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
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
      model.addConstr(B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2 )

  objective = 0
  for j in range(numMutations):
    numZeros = 0
    numOnes = 0
    for i in range(numCells):
      if I[i][j] == 0:
        numZeros += 1
        objective += np.log(beta / (1 - alpha)) * Y[i, j]
      elif I[i][j] == 1:
        numOnes += 1
        objective += np.log((1 - beta) / alpha) * Y[i, j]

    objective += numZeros * np.log(1 - alpha)
    objective += numOnes * np.log(alpha)

  model.setObjective(objective, GRB.MAXIMIZE)
  model.optimize()
  n1 = np.count_nonzero(I)
  n0 = np.count_nonzero(1 - I)
  # print(model.objVal)
  return np.ceil(calcNF(model.objVal, n0, n1, alpha, beta))


def LP_Bounding_direct(I):
  """obj does not need translation"""
  numCells, numMutations = I.shape

  model = Model('LP_3')
  model.Params.OutputFlag = 0
  Y = {}
  for c in range(numCells):
    for m in range(numMutations):
      Y[c, m] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, name='Y({0},{1})'.format(c, m))
  B = {}
  for p in range(numMutations + 1):
    for q in range(numMutations + 1):
      B[p, q, 1, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
                                   name='B[{0},{1},1,1]'.format(p, q))
      B[p, q, 1, 0] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
                                   name='B[{0},{1},1,0]'.format(p, q))
      B[p, q, 0, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
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


  objective = 0
  for j in range(numMutations):
    for i in range(numCells):
      if I[i][j] == 0:
        objective += Y[i, j]
      elif I[i][j] == 1:
        objective += 10 * (1 - Y[i, j])

  model.setObjective(objective, GRB.MINIMIZE)

  model.optimize()

  return model.objVal


def LP_Bounding_direct_4(I):
  numCells, numMutations = I.shape

  model = Model('LP_4')
  model.Params.OutputFlag = 0
  Y = {}
  for c in range(numCells):
    for m in range(numMutations):
      if I[c, m] == 0:
        Y[c, m] = model.addVar(0, 1, obj=1, vtype=GRB.CONTINUOUS, name='Y({0},{1})'.format(c, m))
      elif I[c, m] == 1:
        Y[c, m] = 1

  B = {}
  for p in range(numMutations + 1):
    for q in range(numMutations + 1):
      B[p, q, 1, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
                                   name='B[{0},{1},1,1]'.format(p, q))
      B[p, q, 1, 0] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
                                   name='B[{0},{1},1,0]'.format(p, q))
      B[p, q, 0, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0,
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

  model.optimize()

  # flip(cc, mm, model, Y)

  return np.ceil(model.objVal)


def LP_brief(I):
  model, Y = makeGurobiModel(I)
  return LP_Bounding_Model(model)


def LP_Bounding_Model(model):
  model.optimize()
  return np.int(np.ceil(model.objVal))


if __name__ == '__main__':
  n, m = 8, 8
  x = np.random.randint(2, size=(n, m))

  mm, y = makeGurobiModel(x)


  nf = LP_brief(x)
  nf2 = LP_Bounding_Model(mm)

  print(nf, nf2)
  ind = np.nonzero(1-x)
  a, b = ind[0][0], ind[1][0]

  x[a, b] = 1

  tt1 = time.time()
  nf = LP_brief(x)
  tt1 = time.time() - tt1


  tt2 = time.time()
  flip(a, b, mm, y)
  nf2 = LP_Bounding_Model(mm)
  tt2 = time.time() - tt2

  print(nf, nf2)
  print(tt1, tt2)
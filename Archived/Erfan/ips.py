import os
import subprocess
import numpy as np
import random
import math
from gurobipy import *

def calcNF(obj, n0, n1, alpha = 0.0001, beta = 0.99):
    nom = obj - (n1 * np.log(1 - beta) + n0 * np.log(1 - alpha))
    den = np.log(beta / (1 - alpha))
    return nom / den


def ILP_1(I):
  def nearestInt(x):
    return int(x + 0.5)

  beta, alpha = 0.99, 0.0001
  numCells, numMutations = I.shape
  sol_Y = []
  sol_K = []

  model = Model('ILP_1')
  model.Params.OutputFlag = 0
  Y = {}
  for c in range(numCells):
    for m in range(numMutations):
      Y[c, m] = model.addVar(vtype=GRB.BINARY, name='Y({0},{1})'.format(c, m))
  B = {}
  for p in range(numMutations + 1):
    for q in range(numMutations + 1):
      B[p, q, 1, 1] = model.addVar(vtype=GRB.BINARY, obj=0,
                                   name='B[{0},{1},1,1]'.format(p, q))
      B[p, q, 1, 0] = model.addVar(vtype=GRB.BINARY, obj=0,
                                   name='B[{0},{1},1,0]'.format(p, q))
      B[p, q, 0, 1] = model.addVar(vtype=GRB.BINARY, obj=0,
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

  if model.status == GRB.Status.INFEASIBLE:
    print('The odel is infeasible.')
    exit(0)


  for i in range(numCells):
    sol_Y.append([nearestInt(float(Y[i, j].X)) for j in range(numMutations)])
  # print()
  # print("1: ", model.objVal)
  # n1 = np.count_nonzero(I)
  # n0 = np.count_nonzero(1 - I)
  # print("n0, n1= ", n0, n1)
  # print(calcNF(model.objVal, n0, n1, alpha, beta))
  # print()
  return np.array(sol_Y)


def LP_2(I):
  def nearestInt(x):
    return x
    # return int(x + 0.5)

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

  if model.status == GRB.Status.INFEASIBLE:
    print('The odel is infeasible.')
    exit(0)


  for i in range(numCells):
    sol_Y.append([nearestInt(float(Y[i, j].X)) for j in range(numMutations)])

  final = np.array(sol_Y)
  print()
  print("2: ", model.objVal)
  n1 = np.count_nonzero(I)
  n0 = np.count_nonzero(1 - I)
  print("n0, n1= ", n0, n1)
  print(calcNF(model.objVal, n0, n1, alpha, beta))
  print()
  # print(final)
  return final






if __name__ == '__main__':
  n, m = 8, 8
  x = np.random.randint(2, size=(n, m))

  solution = ILP_1(x)
  fn = len(np.where(solution != x)[0])
  # print(fn)

  print(x)
  print()
  print(solution)
  print()
  solution2 = LP_2(x)
  print(solution2)

  fn2 = len(np.where(solution2 != x)[0])
  print(fn2, fn)
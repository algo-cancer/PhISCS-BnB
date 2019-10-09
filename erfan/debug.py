import numpy as np
from gurobipy import *
import time, copy
import scipy.sparse as sp


if __name__ == '__main__':
  n = 80
  m = 80
  print(n, m)
  x = np.random.randint(2, size=(n, m))
  delta = sp.lil_matrix((n, m))
  # x = np.array([[1, 0, 0, 0, 0, 0],
  #               [1, 1, 0, 0, 0, 1],
  #               [0, 1, 0, 0, 1, 0],
  #               [0, 1, 0, 1, 1, 0],
  #               [0, 0, 0, 0, 0, 0],
  #               [1, 1, 1, 1, 1, 0],
  #               [1, 0, 1, 1, 1, 0],
  #               [1, 0, 1, 1, 0, 1]], dtype=np.int8)
  #
  # delta = np.array([[0, 0, 0, 0, 0, 0],
  #                   [0, 0, 1, 1, 1, 0],
  #                   [1, 0, 0, 0, 0, 0],
  #                   [1, 0, 0, 0, 0, 0],
  #                   [0, 0, 0, 0, 0, 0],
  #                   [0, 0, 0, 0, 0, 0],
  #                   [0, 1, 0, 0, 0, 0],
  #                   [0, 1, 0, 0, 1, 0]], dtype=np.int8)

  xp = x + delta

  I = xp
  continuous = True

  if continuous:
    varType = GRB.CONTINUOUS
  else:
    varType = GRB.BINARY

  numCells, numMutations = I.shape

  model = Model(f'LP_Gurobi_{time.time()}')
  model.Params.OutputFlag = 0
  model.Params.Threads = 1
  Y = {}
  for c in range(numCells):
    for m in range(numMutations):
      if I[c, m] == 0:
        Y[c, m] = model.addVar(0, 1, obj=1, vtype=varType, name='Y({0},{1})'.format(c, m))
      elif I[c, m] == 1:
        Y[c, m] = 1

  B = {}
  for p in range(numMutations):
    for q in range(numMutations):
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

  for p in range(numMutations):
    for q in range(numMutations):
      model.addConstr(B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2)

  model.Params.ModelSense = GRB.MINIMIZE
  model.update()

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

  print("reset here!")
  model.reset()

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

  #######  Change one random coordinate ######
  nnzind = np.nonzero(1 - (x + delta))
  a, b = nnzind[0][0], nnzind[1][0]
  delta[a, b] = 1
  ############################################
  cns = (Y[a, b] == 1)
  model.addConstr(cns)
  cns2 = model.getConstrs()[-1]
  print("add constr")

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

  model.remove(cns2)
  print("remove")

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)


  print("reset here!")
  model.reset()

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

  t = time.time()
  model.optimize()
  t = time.time() -t
  print("optimization time:", t)

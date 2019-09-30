import numpy as np
from interfaces import *
import scipy.sparse as sp
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def blockshaped(arr, nrows, ncols):
  h, w = arr.shape
  assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
  assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
  return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))


def count_flips(I, sol_K, sol_Y):
  flips_0_1 = 0
  flips_1_0 = 0
  flips_2_0 = 0
  flips_2_1 = 0
  n, m = I.shape
  for i in range(n):
      for j in range(m):
          if sol_K[j] == 0:
              if I[i][j] == 0 and sol_Y[i][j] == 1:
                  flips_0_1 += 1
              elif I[i][j] == 1 and sol_Y[i][j] == 0:
                  flips_1_0 += 1
              elif I[i][j] == 2 and sol_Y[i][j] == 0:
                  flips_2_0 += 1
              elif I[i][j] == 2 and sol_Y[i][j] == 1:
                  flips_2_1 += 1
  return flips_0_1


def PhISCS_B(matrix, procnum=0, return_dict={}):
  rc2 = RC2(WCNF())
  n,m = matrix.shape
  par_fnWeight = 1
  par_fpWeight = 10

  Y = np.empty((n,m), dtype=np.int64)
  numVarY = 0
  map_y2ij = {}
  for i in range(n):
      for j in range(m):
          numVarY += 1
          map_y2ij[numVarY] = (i, j)
          Y[i,j] = numVarY

  X = np.empty((n,m), dtype=np.int64)
  numVarX = 0
  for i in range(n):
      for j in range(m):
          numVarX += 1
          X[i,j] = numVarY + numVarX

  B = np.empty((n,m,2,2), dtype=np.int64)
  numVarB = 0
  for p in range(m):
      for q in range(m):
          for i in range(2):
              for j in range(2):
                  numVarB += 1
                  B[p,q,i,j] = numVarY + numVarX + numVarB;
  
  for i in range(n):
      for j in range(m):
          if matrix[i,j] == 0:
              rc2.add_clause([-X[i,j]], weight=par_fnWeight)
              rc2.add_clause([-X[i,j], Y[i,j]])
              rc2.add_clause([X[i,j], -Y[i,j]])
          elif matrix[i,j] == 1:
              rc2.add_clause([-X[i,j]], weight=par_fpWeight)
              rc2.add_clause([X[i,j], Y[i,j]])
              rc2.add_clause([-X[i,j], -Y[i,j]])
          elif matrix[i,j] == 2:
              rc2.add_clause([-X[i,j], Y[i,j]])
              rc2.add_clause([X[i,j], -Y[i,j]])

  for i in range(n):
      for p in range(m):
          for q in range(p, m):
              rc2.add_clause([-Y[i,p], -Y[i,q], B[p,q,1,1]])
              rc2.add_clause([Y[i,p], -Y[i,q], B[p,q,0,1]])
              rc2.add_clause([-Y[i,p], Y[i,q], B[p,q,1,0]])
              rc2.add_clause([-B[p,q,0,1], -B[p,q,1,0], -B[p,q,1,1]])
  
  variables = rc2.compute()

  O = np.empty((n,m), dtype=np.int8)
  numVar = 0
  for i in range(n):
      for j in range(m):
          if matrix[i,j] == 0:
              if variables[numVar] < 0:
                  O[i,j] = 0
              else:
                  O[i,j] = 1
          elif matrix[i,j] == 1:
              if variables[numVar] < 0:
                  O[i,j] = 0
              else:
                  O[i,j] = 1
          elif matrix[i,j] == 2:
              if variables[numVar] < 0:
                  O[i,j] = 0
              else:
                  O[i,j] = 1
          numVar += 1
  
  return_dict[procnum] = count_flips(matrix, matrix.shape[1]*[0], O)[0]
  return count_flips(matrix, matrix.shape[1]*[0], O)


class StaticPhISCSBBounding(BoundingAlgAbstract):
  def __init__(self):
    self.matrix = None

  def reset(self, matrix):
    self.matrix = matrix

  def getBound(self, delta):
    bound = 0
    for block in blockshaped(self.matrix + delta, D.shape[0], 5):
        bound += PhISCS_B(block)
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
  
  algo = StaticPhISCSBBounding()
  xp = np.asarray(x + delta)
  print(algo.getBound(delta))

import sys
sys.path.append('/data/frashidi/Phylogeny_BnB/erfan/')
import numpy as np
from interfaces import *
import scipy.sparse as sp
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


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
  
  flips_0_1 = np.count_nonzero(O-matrix)
  return_dict[procnum] = flips_0_1
  return flips_0_1


class StaticPhISCSBBounding(BoundingAlgAbstract):
  def __init__(self, splitInto=2):
    self.matrix = None
    self.n = None
    self.m = None
    self.splitInto = splitInto

  def reset(self, matrix):
    self.matrix = matrix
    self.n = self.matrix.shape[0]
    self.m = self.matrix.shape[1]

  def getBound(self, delta):
  	# https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    bound = 0
    I = np.array(self.matrix+delta)
    blocks = np.array_split(I, self.splitInto, axis=1)
    for block in blocks:
      bound += PhISCS_B(block)
    return bound + delta.count_nonzero()


if __name__ == '__main__':

  noisy = np.array([
    [0,1,0,0,0,0,1,1,1,0],
    [0,1,1,0,1,1,1,0,1,0],
    [1,0,0,1,0,1,1,1,0,0],
    [1,0,0,0,0,0,0,1,0,0],
    [1,1,1,1,1,1,0,1,0,1],
    [0,1,1,1,1,1,1,1,0,0],
    [1,0,0,1,0,1,0,0,0,0],
    [1,1,1,1,0,0,1,0,1,1],
    [0,0,1,0,1,1,1,1,1,0],
    [1,1,1,1,0,0,1,0,1,1],
  ], dtype=np.int8)
  delta = sp.lil_matrix((noisy.shape))
  delta[0,0] = 1
  
  algo = StaticPhISCSBBounding()
  algo.reset(noisy)
  xp = np.asarray(noisy + delta)
  print(algo.getBound(delta))

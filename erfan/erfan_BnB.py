import numpy as np
from instances import I1
from funcs import *
import pybnb
from utils import *
import operator
from collections import defaultdict
import time
import pandas as pd
from tqdm import tqdm
import itertools
from lp_bounding import makeGurobiModel, flip, unFlipLast, LP_Bounding_Model, LP_brief
from interfaces import *
from Boundings.LP import *
from Boundings.MWM import *
import copy
import scipy.sparse as sp

class ErfanBnB(pybnb.Problem):
  """
  Bounding algorithm:
  - uses gusfield
  - accepts any boundingAlg with the interface
  - I is getting copied
  """
  def __init__(self, I, boundingAlg : BoundingAlgAbstract, checkBounding = False):
    self.I = I
    self.delta = sp.lil_matrix(I.shape, dtype = np.int8) # this can be coo_matrix too
    self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
    self.boundVal = 0
    self.boundingAlg = boundingAlg
    self.boundingAlg.reset(I)
    self.checkBounding = checkBounding


  def getNFlips(self):
    return self.delta.count_nonzero()

  def sense(self):
    return pybnb.minimize

  def objective(self):
    # print(f"OBJ: {self.icf}, {self.getNFlips()}")
    if self.icf:
      return self.getNFlips()
    else:
      return pybnb.Problem.infeasible_objective(self)

  def bound(self):
    if self.checkBounding: # Debugging here
      ll = myPhISCS_I(np.asarray(self.I+self.delta))
      if ll + self.delta.count_nonzero() < self.boundVal:
        print(ll, self.getNFlips(), self.boundVal)
        print(" ============== ")
        print(repr(self.I))
        print(repr(self.delta.todense()))
        print(" ============== ")
        ss = SemiDynamicLPBounding()
        ss.reset(self.I)
        print(f"np.allclose(self.boundingAlg.matrix, self.I)={np.allclose(self.boundingAlg.matrix, self.I)}")
        print(f"len(self.boundingAlg.model.getConstrs()) = {len(self.boundingAlg.model.getConstrs())}")
        print(f"len(ss.model.getConstrs()) = {len(ss.model.getConstrs())}")
        thisAnswer = ss.get_bound(self.delta)
        print(f"np.allclose(self.boundingAlg.matrix, self.I)={np.allclose(self.boundingAlg.matrix, self.I)}")
        print(f"len(self.boundingAlg.model.getConstrs()) = {len(self.boundingAlg.model.getConstrs())}")
        print(f"len(ss.model.getConstrs()) = {len(ss.model.getConstrs())}")

        print(f"{thisAnswer} vs {self.boundingAlg.get_bound(self.delta)} vs {self.boundVal}")
        print(f"{thisAnswer} vs {self.boundingAlg.get_bound(self.delta)} vs {self.boundVal}")

        exit(0)
    return self.boundVal

  def save_state(self, node):
    node.state = (self.delta, self.icf, self.colPair, self.boundVal, self.boundingAlg.get_state())

  def load_state(self, node):
    self.delta, self.icf, self.colPair, self.boundVal, boundingAlgState = node.state
    self.boundingAlg.set_state(boundingAlgState)

  def getCurrentMatrix(self):
    return self.I + self.delta

  def branch(self):
    if self.icf: # by fliping more the objective is not going to get better
      return
    p, q = self.colPair
    p, q, oneone, zeroone, onezero = get_a_coflict(self.getCurrentMatrix(), p, q)

    for a, b in [(onezero, q), (zeroone, p)]:
      # print(f"{(a,b)} is made!")
      node = pybnb.Node()
      nodedelta =  copy.deepcopy(self.delta)
      nodedelta[a, b] = 1
      nodeicf, nodecolPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I + nodedelta )

      # print("--------------", nodedelta.todense())

      newBound = self.boundingAlg.get_bound(nodedelta)
      # pat = np.array([[0, 0, 0, 0, 0],
      #   [0, 0, 0, 0, 0],
      #   [0, 0, 1, 0, 0],
      #   [1, 0, 1, 0, 0],
      #   [0, 0, 0, 0, 0]])
      # if np.allclose(nodedelta.todense() , pat):
      #   print("--------------", newBound, self.boundVal)
      #   ss = SemiDynamicLPBounding()
      #   ss.reset(self.I)
      #   print(ss.getBound(nodedelta), self.boundingAlg.getBound(nodedelta), type(self.boundingAlg))
      #   exit(0)


      nodeboundVal = max(self.boundVal, newBound)
      node.state = (nodedelta, nodeicf, nodecolPair, nodeboundVal, self.boundingAlg.get_state())
      node.queue_priority = - newBound
      yield node

def ErfanBnBSolver(x):
  problem = ErfanBnB(x, EmptyBoundingAlg())
  results = pybnb.solve(problem) #, log=None)
  ans = results.best_node.state[0]
  return ans

if __name__ == '__main__':
  timeLimit = 120
  gurobi_env = Env()
  n, m = 9, 9
  # n, m = 5, 5
  x = np.random.randint(2, size=(n, m))
  # x = np.array([[0, 1, 1, 0],
  #            [1, 0, 1, 1],
  #            [0, 1, 0, 1],
  #            [1, 1, 1, 1]])

# counterexample for (SemiDynamicLPBounding(), 'fifo'),
  # x = np.array([[1, 0, 0, 0, 0, 0],
  #      [1, 1, 0, 0, 0, 1],
  #      [0, 1, 0, 0, 1, 0],
  #      [0, 1, 0, 1, 1, 0],
  #      [0, 0, 0, 0, 0, 0],
  #      [1, 1, 1, 1, 1, 0],
  #      [1, 0, 1, 1, 1, 0],
  #      [1, 0, 1, 1, 0, 1]])

  # x = np.array([[0, 1, 1, 1, 1, 1],
  #                [1, 0, 1, 1, 0, 1],
  #                [1, 0, 1, 1, 0, 1],
  #                [0, 0, 0, 0, 1, 1],
  #                [0, 0, 0, 0, 1, 1],
  #                [1, 0, 1, 1, 0, 0],
  #                [0, 0, 0, 0, 1, 0],
  #                [1, 1, 1, 0, 0, 1]])
  # x = np.array([[1, 0, 1, 0, 0],
  #      [1, 1, 1, 1, 0],
  #      [1, 1, 0, 1, 0],
  #      [0, 1, 0, 1, 1],
  #      [0, 0, 1, 1, 1]], dtype=np.int8)

  # print(repr(x))
  optim = myPhISCS_I(x)
  print("Optimal answer:", optim)
  if optim>25:
    exit(0)

  boundings = [
    # EmptyBoundingAlg(),
    # (NaiveBounding(), 'fifo'), # The time measures of First one is not trusted for cache issues
    # (NaiveBounding(), 'depth'),
    # (NaiveBounding(), 'custom'),
    # (SemiDynamicLPBounding(), 'fifo'),
    # (SemiDynamicLPBounding(), 'depth'),
    (StaticILPBounding(), 'custom'),
    (StaticILPBounding(), 'custom'),
    (SemiDynamicLPBounding(), 'custom'),
    (SemiDynamicLPBounding(), 'custom'),
    (SemiDynamicLPBounding(), 'custom'),
    # (StaticLPBounding(), 'fifo'),
    # (StaticLPBounding(), 'custom'),
    # (StaticLPBounding(), 'custom'),
    # (StaticLPBounding(), 'custom'),
    # (StaticLPBounding(), 'depth'),
    # (DynamicMWMBounding(), 'custom'),
    # (DynamicMWMBounding(), 'custom'),
    # (DynamicMWMBounding(), 'custom'),
    # (DynamicMWMBounding(), 'fifo'),
    # (DynamicMWMBounding(), 'depth'),
    # (DynamicMWMBounding(), 'custom'),
    # (StaticMWMBounding(), 'custom'),
    # (StaticMWMBounding(), 'custom'),
    # (StaticMWMBounding(), 'custom'),
    # (StaticMWMBounding(), 'depth')
  ]

  for boundFunc, queue_strategy in boundings:
    # print(boundFunc.getName(), queue_strategy, flush = True)
  # for boundFunc, queue_strategy in tqdm(boundings):
    time1 = time.time()
    problem1 = ErfanBnB(x, boundFunc, False)
    solver = pybnb.solver.Solver()
    results1 = solver.solve(problem1,  queue_strategy = queue_strategy, log = None, time_limit = timeLimit)
    # results1 = solver.solve(problem1,  queue_strategy = queue_strategy,)
    time1 = time.time() - time1
    delta = results1.best_node.state[0]
    nf1 = delta.count_nonzero()
    # print(repr(delta.todense()))
    print(nf1, str(time1)[:5], results1.nodes, boundFunc.get_name(), queue_strategy, flush=True)


  if False:
    timeLimit = 120
    problem1 = ErfanBnB(x, SemiDynamicLPBounding(), False)
    solver = pybnb.solver.Solver()
    results1 = solver.solve(problem1,  queue_strategy = "custom", log = None, time_limit = timeLimit)
    delta = results1.best_node.state[0]
    nf1 = delta.count_nonzero()
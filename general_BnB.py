from Utils.const import *

from interfaces import *
from ErfanFuncs import *
from Boundings.LP import *
from Boundings.MWM import *


class BnB(pybnb.Problem):
  """
  - Accept Bounding algorithm with the interface
  - uses gusfield if the bounding does not provide next p,q
  - only delta is getting copied
  """
  def __init__(self, I, boundingAlg : BoundingAlgAbstract, checkBounding = False):
    self.I = I
    self.delta = sp.lil_matrix(I.shape, dtype = np.int8) # this can be coo_matrix too
    self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
    self.boundingAlg = boundingAlg
    self.boundingAlg.reset(I)
    self.boundVal = self.boundingAlg.get_bound(self.delta)
    # print(hasattr(self.boundingAlg.yVars[0, 1], "X"))
    self.checkBounding = checkBounding

  def getNFlips(self):
    return self.delta.count_nonzero()

  def sense(self):
    return pybnb.minimize

  def objective(self):
    # print(self.delta.nonzero(), self.boundVal, self.getNFlips())
    # print(f"OBJ: {self.icf}, {self.getNFlips()}")
    if self.icf:
      return self.getNFlips()
    else:
      return pybnb.Problem.infeasible_objective(self)
      # return 1000

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
    # nodes = []
    nf = self.getNFlips() + 1
    for a, b in [(onezero, q), (zeroone, p)]:
      # print(f"{(a,b)} is made!")
      node = pybnb.Node()
      nodedelta =  copy.deepcopy(self.delta)
      nodedelta[a, b] = 1

      # print("line 84:", hasattr(self.boundingAlg.yVars[0, 1], "X"))
      newBound = self.boundingAlg.get_bound(nodedelta)
      # print("line 86:", hasattr(self.boundingAlg.yVars[0, 1], "X"))

      nodeicf, nodecolPair = None, None
      extraInfo = self.boundingAlg.get_extra_info()
      # print(extraInfo)
      if extraInfo is not None:
        if "icf" in extraInfo:
          nodeicf = extraInfo["icf"]
        if "one_pair_of_columns" in extraInfo:
          nodecolPair = extraInfo["one_pair_of_columns"]
      # print(nodeicf, nodecolPair)
      # print(is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I + nodedelta ))
      if nodeicf is None or nodecolPair is None:
        # print("run gusfield")
        nodeicf, nodecolPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I + nodedelta )


      # nodeboundVal = max(self.boundVal, newBound)
      nodeboundVal = newBound
      node.state = (nodedelta, nodeicf, nodecolPair, nodeboundVal, self.boundingAlg.get_state())
      # node.queue_priority = - ( newBound - nf)
      node.queue_priority =  self.boundingAlg.get_priority(newBound - nf, nodeicf)
      # node.queue_priority =  self.boundingAlg.getPriority(nodeboundVal)
      yield node
    #   nodes.append(node)
    # return nodes

# def ErfanBnBSolver(x):
#   problem = BnB(x, EmptyBoundingAlg())
#   results = pybnb.solve(problem) #, log=None)
#   ans = results.best_node.state[0]
#   return ans

if __name__ == '__main__':
  timeLimit = 120

  # n, m = 8, 8
  # n, m = 5, 5
  # x = np.random.randint(2, size=(n, m))
  x = np.array([
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
  ])
  

  print(repr(x))
  optimTime_I = time.time()
  optim = myPhISCS_I(x)
  optimTime_I = time.time() - optimTime_I
  print("Optimal answer (I):", optim)
  print("Optimal time   (I):", optimTime_I)
  optimTime_B = time.time()
  optim = myPhISCS_B(x)
  optimTime_B = time.time() - optimTime_B
  print("Optimal answer (B):", optim)
  print("Optimal time   (B):", optimTime_B)

  if optim>25:
    exit(0)

  boundings = [
    # EmptyBoundingAlg(),
    # (NaiveBounding(), 'fifo'), # The time measures of First one is not trusted for cache issues
    # (NaiveBounding(), 'depth'),
    # (NaiveBounding(), 'custom'),
    # (RandomPartitioning(ascendingOrder=True), 'custom'),
    # (SemiDynamicLPBounding(), 'fifo'),
    # (SemiDynamicLPBounding(), 'depth'),
    # (StaticILPBounding(), 'custom'),
    # (StaticILPBounding(), 'custom'),
    # (SemiDynamicLPBounding(), 'custom'),
    # (SemiDynamicLPBounding(), 'custom'),
    # (SemiDynamicLPBounding(), 'custom'),
    (SemiDynamicLPBounding(ratio=None, continuous = True), 'custom'),
    # (SemiDynamicLPBounding(ratio=None, continuous = False), 'custom'),
    # (StaticLPBounding(), 'fifo'),
    # (StaticLPBounding(), 'custom'),
    # (StaticLPBounding(), 'custom'),
    # (StaticLPBounding(), 'custom'),
    # (StaticLPBounding(), 'depth'),
    (DynamicMWMBounding(), 'custom'),
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
    time1 = time.time()
    problem1 = BnB(x, boundFunc, False)
    solver = pybnb.solver.Solver()
    results1 = solver.solve(problem1,  queue_strategy = queue_strategy, log = None, time_limit = timeLimit)
    # results1 = solver.solve(problem1,  queue_strategy = queue_strategy,)
    time1 = time.time() - time1
    delta = results1.best_node.state[0]
    nf1 = delta.count_nonzero()
    print(nf1, str(time1)[:5], results1.nodes, boundFunc.get_name(), queue_strategy, flush=True)


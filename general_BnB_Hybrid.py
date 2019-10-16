from Utils.const import *

from interfaces import *
from ErfanFuncs import *
from Boundings.LP import *
from Boundings.MWM import *

CHANGE_BOUNDING = 18

class BnB(pybnb.Problem):
  """
  - Accept Bounding algorithm with the interface
  - uses gusfield if the bounding does not provide next p,q
  - only delta is getting copied
  """
  def __init__(self, I, boundingAlg1 : BoundingAlgAbstract, boundingAlg2 : BoundingAlgAbstract):
    self.I = I
    self.delta = sp.lil_matrix(I.shape, dtype = np.int8) # this can be coo_matrix too
    self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
    self.boundingAlg1 = boundingAlg1
    self.boundingAlg2 = boundingAlg2
    self.boundingAlg1.reset(I)
    self.boundingAlg2.reset(I)
    self.boundVal = self.boundingAlg1.getBound(self.delta)
    self.boundingAlg2.getBound(self.delta)

  def getNFlips(self):
    return self.delta.count_nonzero()

  def sense(self):
    return pybnb.minimize

  def objective(self):
    if self.icf:
      return self.getNFlips()
    else:
      return pybnb.Problem.infeasible_objective(self)

  def bound(self):
    return self.boundVal

  def save_state(self, node):
    if self.getNFlips() < CHANGE_BOUNDING:
      node.state = (self.delta, self.icf, self.colPair, self.boundVal, self.boundingAlg1.getState())
    else:
      node.state = (self.delta, self.icf, self.colPair, self.boundVal, self.boundingAlg2.getState())

  def load_state(self, node):
    self.delta, self.icf, self.colPair, self.boundVal, boundingAlgState = node.state
    if self.getNFlips() < CHANGE_BOUNDING:
      self.boundingAlg1.setState(boundingAlgState)
    else:
      self.boundingAlg2.setState(boundingAlgState)

  def getCurrentMatrix(self):
    return self.I + self.delta

  def branch(self):
    if self.icf: # by fliping more the objective is not going to get better
      return
    p, q = self.colPair
    p, q, oneone, zeroone, onezero = get_a_coflict(self.getCurrentMatrix(), p, q)
    nf = self.getNFlips() + 1
    for a, b in [(onezero, q), (zeroone, p)]:
      node = pybnb.Node()
      nodedelta =  copy.deepcopy(self.delta)
      nodedelta[a, b] = 1


      if nf < CHANGE_BOUNDING:
        newBound = self.boundingAlg1.getBound(nodedelta)
        extraInfo = self.boundingAlg1.getExtraInfo()
      else:
        newBound = self.boundingAlg2.getBound(nodedelta)
        extraInfo = self.boundingAlg2.getExtraInfo()
      
      nodeicf, nodecolPair = None, None
      if extraInfo is not None:
        if "icf" in extraInfo:
          nodeicf = extraInfo["icf"]
        if "one_pair_of_columns" in extraInfo:
          nodecolPair = extraInfo["one_pair_of_columns"]
      if nodeicf is None or nodecolPair is None:
        nodeicf, nodecolPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I + nodedelta )

      nodeboundVal = newBound
      if nf < CHANGE_BOUNDING:
        node.state = (nodedelta, nodeicf, nodecolPair, nodeboundVal, self.boundingAlg1.getState())
        node.queue_priority =  self.boundingAlg1.getPriority(newBound - nf, nodeicf)
      else:
        node.state = (nodedelta, nodeicf, nodecolPair, nodeboundVal, self.boundingAlg2.getState())
        node.queue_priority =  self.boundingAlg2.getPriority(newBound - nf, nodeicf)
      yield node

if __name__ == '__main__':
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

  time1 = time.time()
  b1 = SemiDynamicLPBounding(ratio=None, continuous = True)
  b2 = DynamicMWMBounding()
  problem1 = BnB(x, b1, b2)
  solver = pybnb.solver.Solver()
  results1 = solver.solve(problem1, queue_strategy='custom', log = None)
  time1 = time.time() - time1
  delta = results1.best_node.state[0]
  nf1 = delta.count_nonzero()
  print(nf1, str(time1)[:5], results1.nodes, flush=True)

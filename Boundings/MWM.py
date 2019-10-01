import sys
if __name__ == '__main__':
  sys.path.append('../Utils')
from const import *
from interfaces import *
from ErfanFuncs import *

class DynamicMWMBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None, ascendingOrder = False):
    """
    :param ratio:
    :param ascendingOrder: if True the column pair with max weight is chosen in extra info
    """
    self.matrix = None
    self.G = None
    self._extraInfo = {}
    self.ascendingOrder = ascendingOrder
    self.ratio = ratio

  def reset(self, matrix):
    self.matrix = matrix
    self.G = nx.Graph()
    for p in range(self.matrix.shape[1]):
      for q in range(p + 1, self.matrix.shape[1]):
        self.calc_min0110_for_one_pair_of_columns(p, q, self.matrix)

  def getExtraInfo(self):
    # return None
    return self._extraInfo

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
    self.extraInfo = None

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

    sign = 1 if self.ascendingOrder else -1

    optPairValue = delta.shape[0] * delta.shape[1] * (-sign) # either + inf or - inf
    optPair = None
    lb = 0
    for a, b in best_pairing:
      lb += self.G[a][b]["weight"]
      if self.G[a][b]["weight"] * sign > optPairValue * sign:
        optPairValue = self.G[a][b]["weight"] * (-sign)
        optPair = (a, b)
    self.G = oldG
    self._extraInfo = {"icf": (lb == 0), "one_pair_of_columns": optPair if lb > 0 else None}
    return lb + flipsMat.shape[0]


class StaticMWMBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None, ascendingOrder = False):
    """
    :param ratio:
    :param ascendingOrder: if True the column pair with max weight is chosen in extra info
    """
    self.ratio = ratio
    self.matrix = None
    self.ascendingOrder = ascendingOrder
    self._extraInfo = {}

  def reset(self, matrix):
    self.matrix = matrix

  def getExtraInfo(self):
    return self._extraInfo

  def getBound(self, delta):
    self.extraInfo = None
    nFlips = delta.count_nonzero()
    currentMatrix = self.matrix + delta
    self.G = nx.Graph()
    for p in range(currentMatrix.shape[1]):
      for q in range(p + 1, currentMatrix.shape[1]):
        self.calc_min0110_for_one_pair_of_columns(p, q, currentMatrix)
    best_pairing = nx.max_weight_matching(self.G)

    sign = 1 if self.ascendingOrder else -1

    optPairValue = delta.shape[0] * delta.shape[1] * (-sign) # either + inf or - inf
    optPair = None
    lb = 0
    for a, b in best_pairing:
      lb += self.G[a][b]["weight"]
      if self.G[a][b]["weight"] * sign > optPairValue * sign:
        optPairValue = self.G[a][b]["weight"]
        optPair = (a, b)

    self._extraInfo = {"icf": (lb == 0), "one_pair_of_columns": optPair if lb > 0 else None}

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


  staticMWMBounding = StaticMWMBounding()
  staticMWMBounding.reset(x)

  algo = DynamicMWMBounding()
  resetTime = time.time()
  algo.reset(x)
  resetTime = time.time() - resetTime
  print(resetTime)

  # print(delta.count_nonzero())
  # print(x+delta)
  print(algo.getBound(delta), staticMWMBounding.getBound(delta))
  print(myPhISCS_B(np.array(x + delta)))

  for t in range(100):
    ind = np.nonzero(1 - (x+delta))
    if ind[0].shape[0] == 0:
      print("DONE!")
      break
    a, b = ind[0][0], ind[1][0]
    delta[a, b] = 1

    # algo.reset(x)
    calcTime = time.time()
    bndAdapt = algo.getBound(delta)
    calcTime = time.time() - calcTime
    print(calcTime)
    print(algo.getExtraInfo())
    staticMWMBoundingBnd = staticMWMBounding.getBound(delta)
    print( bndAdapt == staticMWMBoundingBnd, bndAdapt, staticMWMBoundingBnd)

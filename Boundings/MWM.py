import sys
if __name__ == '__main__':
  sys.path.append('../Utils')
from const import *
from interfaces import *
from ErfanFuncs import *


class RandomPartitioning(BoundingAlgAbstract):
  def __init__(self, ratio=None, ascendingOrder=False):
    """
    :param ratio:
    :param ascendingOrder: if True the column pair with max weight is chosen in extra info
    :param randomPartitioning: if False best partitioning is used, otherwise a random
    """
    self.matrix = None
    self._extraInfo = None
    self.ascendingOrder = ascendingOrder
    self.ratio = ratio
    self.dist = None

  def getName(self):
    return type(self).__name__+f"_{self.ratio}_{self.ascendingOrder}"

  def getExtraInfo(self):
    return self._extraInfo

  def reset(self, matrix):
    self.matrix = matrix
    self.dist = np.zeros(tuple([matrix.shape[1]] * 2), dtype = np.int)
    for i in range(self.dist.shape[0]):
      for j in range(i):
        self.dist[i, j] = self.costOfColPair(i, j, self.matrix)
        self.dist[j, i] = self.dist[i, j]

  def getBound(self, delta):
    self._extraInfo = None

    currentMatrix = self.matrix + delta
    flipsMat = np.transpose(delta.nonzero())
    flippedColsSet = set(flipsMat[:, 1])

    d = int(self.matrix.shape[1] / 2)
    partitions_id = np.random.choice(range(self.matrix.shape[1]), size=(d, 2), replace=False)
    lb = 0
    sign = 1 if self.ascendingOrder else -1

    optPairValue = delta.shape[0] * delta.shape[1] * (-sign)  # either + inf or - inf
    optPair = None

    for x in partitions_id:
      costOfX = None
      if x[0] not in flippedColsSet and x[1] not in flippedColsSet:
        costOfX = self.dist[x[0], x[1]]
      else:
        costOfX = self.costOfColPair(x[0], x[1], currentMatrix)
      lb+= costOfX
      if costOfX * sign > optPairValue * sign and costOfX > 0:
        optPairValue = costOfX
        optPair = x
    if lb > 0 and True:
      self._extraInfo = {"icf": False, "one_pair_of_columns": (optPair[0], optPair[1])}
    return lb + flipsMat.shape[0]

  def costOfColPair(self, p, q, mat):
    foundOneOne = False
    numberOfZeroOne = 0
    numberOfOneZero = 0
    for r in range(mat.shape[0]):
      if mat[r, p] == 1 and mat[r, q] == 1:
        foundOneOne = True
      if mat[r, p] == 0 and mat[r, q] == 1:
        numberOfZeroOne += 1
      if mat[r, p] == 1 and mat[r, q] == 0:
        numberOfOneZero += 1
    if foundOneOne:
      if numberOfZeroOne * numberOfOneZero > 0:
        return min(numberOfZeroOne, numberOfOneZero)
    return 0


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

  def getName(self):
    return type(self).__name__+f"_{self.ratio}_{self.ascendingOrder}"

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
    self_extraInfo = None

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
      if self.G[a][b]["weight"] * sign > optPairValue * sign and self.G[a][b]["weight"] > 0:
        optPairValue = self.G[a][b]["weight"]
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

  def getName(self):
    return type(self).__name__+f"_{self.ratio}_{self.ascendingOrder}"

  def reset(self, matrix):
    self.matrix = matrix

  def getExtraInfo(self):
    return self._extraInfo

  def getBound(self, delta):
    self._extraInfo = None
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
      if self.G[a][b]["weight"] * sign > optPairValue * sign and self.G[a][b]["weight"] > 0:
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
  n, m = 10, 5
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

  algs = [
    StaticMWMBounding(),
    DynamicMWMBounding(ascendingOrder = False),
    DynamicMWMBounding(ascendingOrder = True),
    RandomPartitioning(ascendingOrder = False),
    RandomPartitioning(ascendingOrder = True),
  ]
  print("optimal:", myPhISCS_B(np.array(x + delta)))
  for algo in algs:
    resetTime = time.time()
    algo.reset(x)
    resetTime = time.time() - resetTime
    print(algo.getName(), algo.getBound(delta), resetTime)

  # exit(0)
  for t in range(3):
    ind = np.nonzero(1 - (x+delta))
    if ind[0].shape[0] == 0:
      print("DONE!")
      break
    a, b = ind[0][0], ind[1][0]
    delta[a, b] = 1
    print("t =", t)
    for algo in algs:
      print(algo.getName(), algo.getBound(delta), algo.getExtraInfo())

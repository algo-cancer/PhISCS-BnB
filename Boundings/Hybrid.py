import sys
from Utils.const import *
from Utils.interfaces import *
from . import LP
from . import MWM


class HybridBounding(BoundingAlgAbstract):
  def __init__(self, firstBounding=None, secondBounding=True, ratioNFlips=None):
    """
    :param firstBounding:
    :param secondBounding:
    :param ratioNFlips:
    """
    self.matrix = None
    self.ratioNFlips = ratioNFlips
    self.firstBounding = firstBounding
    self.secondBounding = secondBounding
    self.times = None

  def getName(self):
    return type(self).__name__+f"_{self.firstBounding.getName()}_{self.secondBounding.getName()}_{self.ratioNFlips}"

  def reset(self, matrix):
    self.times = {"modelPreperationTime": 0, "optimizationTime": 0,}
    self.firstBounding.reset(matrix)
    self.secondBounding.reset(matrix)

  def getBound(self, delta):
    flips = delta.count_nonzero()
    if flips < self.ratioNFlips:
      bound = self.firstBounding.getBound(delta)
    else:
      bound = self.secondBounding.getBound(delta)
    return bound


if __name__ == '__main__':

  n, m = 15, 15
  x = np.random.randint(2, size=(n, m))
  delta = sp.lil_matrix((n, m ))
  delta[0,0] = 1

  b1 = LP.SemiDynamicLPBounding(ratio=None, continuous=True)
  b2 = MWM.DynamicMWMBounding()
  hybridBounding = HybridBounding(firstBounding=b1, secondBounding=b2, ratioNFlips=5)
  hybridBounding.reset(x)
  print(hybridBounding.getBound(delta))

  print(b1.getBound(delta))
  print(b2.getBound(delta))


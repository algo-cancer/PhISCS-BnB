import scipy.sparse as sp
import numpy as np


class BoundingAlgAbstract:
  def __init__(self):
    pass

  def reset(self, matrix):
    raise NotImplementedError("The method not implemented")

  def getBound(self, delta):
    """
    This bound should include the flips done so far too
    delta: a sparse matrix with fliped ones
    """
    raise NotImplementedError("The method not implemented")

  def getName(self):
    return type(self).__name__

  def getState(self):
    return None

  def setState(self, state):
    assert state is None
    pass


class NaiveBounding(BoundingAlgAbstract):
  def __init__(self):
    pass

  def reset(self, matrix):
    pass

  def getBound(self, delta):
    return delta.count_nonzero()



if __name__ == '__main__':
  a = BoundingAlgAbstract()
  print(a.getName())

  a = NaiveBounding()
  print(a.getName())

  sparseZero = sp.lil_matrix((10, 10), dtype = np.int8)
  sparseZero[2, 2] = 1
  sparseZero[2, 3] = 1
  print(a.getBound(sparseZero))
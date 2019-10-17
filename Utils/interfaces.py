import scipy.sparse as sp
import numpy as np


class BoundingAlgAbstract:
  def __init__(self):
    pass

  def reset(self, matrix):
    raise NotImplementedError("The method not implemented")

  def get_bound(self, delta):
    """
    This bound should include the flips done so far too
    delta: a sparse matrix with fliped ones
    """
    raise NotImplementedError("The method not implemented")

  def get_name(self):
    return type(self).__name__

  def get_state(self):
    return None

  def set_state(self, state):
    assert state is None
    pass


  def get_extra_info(self):
    """
    Some bounding algorithms can provide extra information after calling bounding.
    E.g.,
    return {"icf":True, "bestPair":(a,b)}
    """
    return set()

  def get_priority(self, new_bound, icf=False):
    return -new_bound


class NaiveBounding(BoundingAlgAbstract):
  def __init__(self):
    super().__init__()

  def reset(self, matrix):
    pass

  def get_bound(self, delta):
    return delta.count_nonzero()


if __name__ == '__main__':
  a = BoundingAlgAbstract()
  print(a.get_name())

  a = NaiveBounding()
  print(a.get_name())

  sparse_zero = sp.lil_matrix((10, 10), dtype=np.int8)
  sparse_zero[2, 2] = 1
  sparse_zero[2, 3] = 1
  print(a.get_bound(sparse_zero))
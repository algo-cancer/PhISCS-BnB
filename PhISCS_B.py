import numpy as np
from interfaces import *
import scipy.sparse as sp


def blockshaped(arr, nrows, ncols):
  h, w = arr.shape
  assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
  assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
  return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

def lb_phiscs_b(D, a, b):
    lb = 0
    for block in blockshaped(D, D.shape[0], 5):
        solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1), c_time = PhISCS_B(block)
        lb += flips_0_1

    # blocks = blockshaped(D, D.shape[0], 5)
    # with Pool(processes=len(blocks)) as pool:
    #     result = pool.map(PhISCS_B, blocks)
    # for x in result:
    #     lb += x[1][0]
    
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return lb, {}, best_pair_qp, icf


class StaticPhISCSBBounding(BoundingAlgAbstract):
  def __init__(self):
    self.matrix = None


  def reset(self, matrix):
    self.matrix = matrix

  def getBound(self, delta):
    model, Y = StaticLPBounding.makeGurobiModel(self.matrix + delta, continuous= False)
    optim = StaticLPBounding.LP_Bounding_From_Model(model)
    if self.ratio is not None:
      bound = np.int(np.ceil(self.ratio * optim))
    else:
      bound = np.int(np.ceil(optim))
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
  ss = StaticLPBounding()
  ss.reset(x)

  algo = SemiDynamicLPBounding()
  resetTime = time.time()
  algo.reset(x)
  resetTime = time.time() - resetTime
  print(resetTime)

  xp = np.asarray(x + delta)
  print(type(xp))
  optim = myPhISCS_I(xp)
  print("Optimal answer:", optim)
  print(len(algo.model.getConstrs()))
  print(StaticLPBounding.LP_brief(xp), algo.getBound(delta), ss.getBound(delta))
  print(len(algo.model.getConstrs()))

  for _ in range(5):
    print(ss.getBound(delta))
  exit(0)
  for t in range(3):
    ind = np.nonzero(1 - (x+delta))
    a, b = ind[0][0], ind[1][0]
    delta[a, b] = 1

    calcTime = time.time()
    bndAdapt = algo.getBound(delta)
    calcTime = time.time() - calcTime
    print(calcTime)
    bndFull = StaticLPBounding.LP_brief(x+delta) + t + 1
    print(bndFull == bndAdapt, bndFull, bndAdapt)
    ssBnd = ss.getBound(delta)
    print(bndFull == ssBnd, ssBnd)

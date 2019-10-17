from Utils.const import *
from Utils.util import *
from Utils.instances import *


def myPhISCS_B(x):
  solution, (f_0_1_b, f_1_0_b, f_2_0_b, f_2_1_b), cb_time = PhISCS_B(x, beta=0.90, alpha=0.00000001)
  nf = len(np.where(solution != x)[0])
  return nf


def myPhISCS_I(x):
  solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1), ci_time = PhISCS_I(x, beta=0.90, alpha=0.00000001)
  nf = len(np.where(solution != x)[0])
  return nf


def is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I):
  def sort_bin(a):
    b = np.transpose(a)
    b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
    idx = np.argsort(b_view.ravel())[::-1]
    c = b[idx]
    return np.transpose(c), idx

  O, idx = sort_bin(I)
  # todo: delete duplicate columns
  # print(O, '\n')
  Lij = np.zeros(O.shape, dtype=int)
  for i in range(O.shape[0]):
    maxK = 0
    for j in range(O.shape[1]):
      if O[i, j] == 1:
        Lij[i, j] = maxK
        maxK = j + 1
  # print(Lij, '\n')
  Lj = np.amax(Lij, axis=0)
  # print(Lj, '\n')

  for i in range(O.shape[0]):
    for j in range(O.shape[1]):
      if O[i, j] == 1:
        if Lij[i, j] != Lj[j]:
          return False, (idx[j], idx[Lj[j] - 1])
  return True, (None, None)


def get_a_coflict(D, p, q):
  # todo: oneone is not important you can get rid of
  oneone = None
  zeroone = None
  onezero = None
  for r in range(D.shape[0]):
    if D[r, p] == 1 and D[r, q] == 1:
      oneone = r
    if D[r, p] == 0 and D[r, q] == 1:
      zeroone = r
    if D[r, p] == 1 and D[r, q] == 0:
      onezero = r
    if oneone != None and zeroone != None and onezero != None:
      return (p, q, oneone, zeroone, onezero)
  return None

def get_lower_bound_new(noisy, partition_randomly=False):
  def get_important_pair_of_columns_in_conflict(D):
    important_columns = defaultdict(lambda: 0)
    for p in range(D.shape[1]):
      for q in range(p + 1, D.shape[1]):
        oneone = 0
        zeroone = 0
        onezero = 0
        for r in range(D.shape[0]):
          if D[r, p] == 1 and D[r, q] == 1:
            oneone += 1
          if D[r, p] == 0 and D[r, q] == 1:
            zeroone += 1
          if D[r, p] == 1 and D[r, q] == 0:
            onezero += 1
        ## greedy approach based on the number of conflicts in a pair of columns
        # if oneone*zeroone*onezero > 0:
        #     important_columns[(p,q)] += oneone*zeroone*onezero
        ## greedy approach based on the min number of 01 or 10 in a pair of columns
        if oneone > 0:
          important_columns[(p, q)] += min(zeroone, onezero)
    return important_columns

  def get_partition_sophisticated(D):
    ipofic = get_important_pair_of_columns_in_conflict(D)
    if len(ipofic) == 0:
      return []
    sorted_ipofic = sorted(ipofic.items(), key=operator.itemgetter(1), reverse=True)
    pairs = [sorted_ipofic[0][0]]
    elements = [sorted_ipofic[0][0][0], sorted_ipofic[0][0][1]]
    sorted_ipofic.remove(sorted_ipofic[0])
    for x in sorted_ipofic[:]:
      notFound = True
      for y in x[0]:
        if y in elements:
          sorted_ipofic.remove(x)
          notFound = False
          break
      if notFound:
        pairs.append(x[0])
        elements.append(x[0][0])
        elements.append(x[0][1])
    # print(sorted_ipofic, pairs, elements)
    partitions = []
    for x in pairs:
      partitions.append(D[:, x])
    return partitions

  def get_partition_random(D):
    d = int(D.shape[1] / 2)
    partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
    partitions = []
    for x in partitions_id:
      partitions.append(D[:, x])
    return partitions

  def get_lower_bound_for_a_pair_of_columns(D):
    foundOneOne = False
    numberOfZeroOne = 0
    numberOfOneZero = 0
    for r in range(D.shape[0]):
      if D[r, 0] == 1 and D[r, 1] == 1:
        foundOneOne = True
      if D[r, 0] == 0 and D[r, 1] == 1:
        numberOfZeroOne += 1
      if D[r, 0] == 1 and D[r, 1] == 0:
        numberOfOneZero += 1
    if foundOneOne:
      if numberOfZeroOne * numberOfOneZero > 0:
        return min(numberOfZeroOne, numberOfOneZero)
    return 0

  LB = []
  if partition_randomly:
    partitions = get_partition_random(noisy)
  else:
    partitions = get_partition_sophisticated(noisy)
  for D in partitions:
    LB.append(get_lower_bound_for_a_pair_of_columns(D))
  return sum(LB)


def get_lower_bound(noisy, partition_randomly=False):
  def get_important_pair_of_columns_in_conflict(D):
    important_columns = defaultdict(lambda: 0)
    for p in range(D.shape[1]):
      for q in range(p + 1, D.shape[1]):
        oneone = 0
        zeroone = 0
        onezero = 0
        for r in range(D.shape[0]):
          if D[r, p] == 1 and D[r, q] == 1:
            oneone += 1
          if D[r, p] == 0 and D[r, q] == 1:
            zeroone += 1
          if D[r, p] == 1 and D[r, q] == 0:
            onezero += 1
        if oneone * zeroone * onezero > 0:
          important_columns[(p, q)] += oneone * zeroone * onezero
    return important_columns

  def get_partition_sophisticated(D):
    ipofic = get_important_pair_of_columns_in_conflict(D)
    if len(ipofic) == 0:
      return []
    sorted_ipofic = sorted(ipofic.items(), key=operator.itemgetter(1), reverse=True)
    pairs = [sorted_ipofic[0][0]]
    elements = [sorted_ipofic[0][0][0], sorted_ipofic[0][0][1]]
    sorted_ipofic.remove(sorted_ipofic[0])
    for x in sorted_ipofic[:]:
      notFound = True
      for y in x[0]:
        if y in elements:
          sorted_ipofic.remove(x)
          notFound = False
          break
      if notFound:
        pairs.append(x[0])
        elements.append(x[0][0])
        elements.append(x[0][1])
    # print(sorted_ipofic, pairs, elements)
    partitions = []
    for x in pairs:
      partitions.append(D[:, x])
    return partitions

  def get_partition_random(D):
    d = int(D.shape[1] / 2)
    partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
    partitions = []
    for x in partitions_id:
      partitions.append(D[:, x])
    return partitions

  def get_lower_bound_for_a_pair_of_columns(D):
    foundOneOne = False
    numberOfZeroOne = 0
    numberOfOneZero = 0
    for r in range(D.shape[0]):
      if D[r, 0] == 1 and D[r, 1] == 1:
        foundOneOne = True
      if D[r, 0] == 0 and D[r, 1] == 1:
        numberOfZeroOne += 1
      if D[r, 0] == 1 and D[r, 1] == 0:
        numberOfOneZero += 1
    if foundOneOne:
      if numberOfZeroOne * numberOfOneZero > 0:
        return min(numberOfZeroOne, numberOfOneZero)
    return 0

  LB = []
  if partition_randomly:
    partitions = get_partition_random(noisy)
  else:
    partitions = get_partition_sophisticated(noisy)
  for D in partitions:
    LB.append(get_lower_bound_for_a_pair_of_columns(D))
  return sum(LB)


def getMatrixHash(x):
  return hash(x.tostring()) % 10000000


if __name__ == '__main__':
  I1 = np.array( [[0,1,1,0]
,[1,1,0,1]
,[1,1,1,0]
,[0,0,1,0]])
  print(myPhISCS_I(I1), myPhISCS_B(I1))
  # print(is_conflict_free_gusfield(I1))
  # for i in [1]: #range(100):
  #   np.random.seed(i)
  #   xt = get_lower_bound(I1, True)
  #   xf = get_lower_bound(I1, False)
  #   if xt>1:
  #     print(xt, xf, i)
  # # for I in [I1, I2, I3, I4, I5]:
  # #   res = is_conflict_free_gusfield(I)
  # #   print(res)
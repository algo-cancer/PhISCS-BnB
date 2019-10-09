import sys
if __name__ == '__main__':
  sys.path.append('../Utils')
  from const import *
elif "constHasRun" not in globals():
  from Utils.const import *
from ErfanFuncs import myPhISCS_I
from interfaces import *

class DynamicLPBounding(BoundingAlgAbstract):
  def __init__(self, ratio=None):
    raise NotImplementedError("The method not implemented")


class SemiDynamicLPBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None, continuous = True, nThreads = 1, tool = "Gurobi", prioritySign = -1):
    """
    :param ratio:
    :param continuous:
    :param nThreads:
    :param tool: in ["Gurobi", "ORTools"]
    """
    self.ratio = ratio
    self.matrix = None
    self.model = None
    self.yVars = None
    self.continuous = continuous
    self.nThreads = nThreads
    self.tool = tool
    self.times = None
    self.prioritySign = prioritySign

  def getName(self):
    return type(self).__name__+f"_{self.ratio}_{self.continuous}_{self.tool}_{self.prioritySign}"

  def reset(self, matrix):
    self.times = {"modelPreperationTime": 0, "optimizationTime": 0,}
    self.matrix = matrix

    modelTime = time.time()
    if self.tool == "Gurobi":
      self.model, self.yVars = StaticLPBounding.makeGurobiModel(self.matrix, continuous=self.continuous)
    elif self.tool == "ORTools":
      self.model, self.yVars = StaticLPBounding.makeORToolsModel(self.matrix, continuous=self.continuous)
    modelTime = time.time() - modelTime
    self.times["modelPreperationTime"] += modelTime

    optTime = time.time()
    if self.tool == "Gurobi":
      self.model.optimize()
    elif self.tool == "ORTools":
      self.model.Solve()
    optTime = time.time() - optTime
    self.times["optimizationTime"] += optTime
    # print("First Optimize:", optTime)


  def _flip(self, c, m):
    self.model.addConstr(self.yVars[c, m] == 1)

  def _unFlipLast(self):
    self.model.remove(self.model.getConstrs()[-1])

  def getBound(self, delta):
    # print(self.yVars[0,0].X)
    # self._extraInfo = None
    flips = np.transpose(delta.nonzero())

    modelTime = time.time()
    newConstrs = (self.yVars[flips[i, 0], flips[i, 1]] == 1 for i in range(flips.shape[0]))
    if self.tool == "Gurobi":
      newConstrsReturned = self.model.addConstrs(newConstrs)
      # self.model.update()
    elif self.tool == "ORTools":
      for constrant in newConstrs:
        self.model.Add(constrant)
    modelTime = time.time() - modelTime
    self.times["modelPreperationTime"] += modelTime

    objVal = None
    # self.model.reset()
    optTime = time.time()
    if self.tool == "Gurobi":
      self.model.optimize()
      objVal = np.int(np.ceil(self.model.objVal))
    elif self.tool == "ORTools":
      self.model.Solve()
      objVal = self.model.Objective().Value()
    optTime = time.time() - optTime
    self.times["optimizationTime"] += optTime
    # print("otptime in getBound:", optTime)



    if self.ratio is not None:
      bound = np.int(np.ceil(self.ratio * objVal))
    else:
      bound = np.int(np.ceil(objVal))

    modelTime = time.time()
    for cnstr in newConstrsReturned.values():
      self.model.remove(cnstr)
    # self.model.update()
    modelTime = time.time() - modelTime
    self.times["modelPreperationTime"] += modelTime

    return bound

  def hasState(self):
    for i in range(self.matrix.shape[0]):
      for j in range(self.matrix.shape[1]):
        if not isinstance(self.yVars[i, j], np.int):
          return hasattr(self.yVars[i, j], "X")

  def getPriority(self, newBound, icf = False):
    if icf:
      return 1000
    else:
      return newBound * self.prioritySign


class StaticLPBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None, continuous = True):
    self.ratio = ratio
    self.matrix = None
    self.continuous = continuous


  def getName(self):
    return type(self).__name__+f"_{self.ratio}_{self.continuous}"


  def reset(self, matrix):
    self.matrix = matrix

  def getBound(self, delta):
    bound = StaticLPBounding.LP_brief(self.matrix + delta, self.continuous)
    if self.ratio is not None:
      bound = np.int(np.ceil(self.ratio * bound))
    else:
      bound = np.int(np.ceil(bound))

    return bound + delta.count_nonzero()

  @staticmethod
  def LP_brief(I, continuous= True):
    model, Y = StaticLPBounding.makeGurobiModel(I, continuous = continuous)
    return StaticLPBounding.LP_Bounding_From_Model(model)

  @staticmethod
  def LP_Bounding_From_Model(model):
    model.optimize()
    return np.int(np.ceil(model.objVal))

  @staticmethod
  def makeORToolsModel(I, continuous = True):
    model = pywraplp.Solver(f'LP_ORTools_{time.time()}', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    numCells = I.shape[0]
    numMutations = I.shape[1]
    Y = {}
    numOne = 0
    for c in range(numCells):
        for m in range(numMutations):
            if I[c, m] == 0:
                Y[c, m] = model.NumVar(0, 1, 'Y({0},{1})'.format(c, m))
            elif I[c, m] == 1:
                numOne += 1
                Y[c, m] = 1
    B = {}
    for p in range(numMutations):
        for q in range(numMutations):
            B[p, q, 1, 1] = model.NumVar(0, 1, 'B[{0},{1},1,1]'.format(p, q))
            B[p, q, 1, 0] = model.NumVar(0, 1, 'B[{0},{1},1,0]'.format(p, q))
            B[p, q, 0, 1] = model.NumVar(0, 1, 'B[{0},{1},0,1]'.format(p, q))
    for i in range(numCells):
        for p in range(numMutations):
            for q in range(numMutations):
                model.Add(Y[i,p] + Y[i,q] - B[p,q,1,1] <= 1)
                model.Add(-Y[i,p] + Y[i,q] - B[p,q,0,1] <= 0)
                model.Add(Y[i,p] - Y[i,q] - B[p,q,1,0] <= 0)
    for p in range(numMutations):
        for q in range(numMutations):
            model.Add(B[p,q,0,1] + B[p,q,1,0] + B[p,q,1,1] <= 2)

    objective = sum([Y[c, m] for c in range(numCells) for m in range(numMutations)])
    model.Minimize(objective)
    return model, Y
    # b = time.time()
    # result_status = model.Solve()
    # c = time.time()
    # optimal_solution = model.Objective().Value()
    # lb = np.int(np.ceil(optimal_solution)) - numOne
    #
    # icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
    # d = time.time()
    # t1 = b-a
    # t2 = c-b
    # t3 = d-c

  @staticmethod
  def makeGurobiModel(I, continuous = True):
    if continuous:
      varType = GRB.CONTINUOUS
    else:
      varType = GRB.BINARY

    numCells, numMutations = I.shape

    model = Model(f'LP_Gurobi_{time.time()}')
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    Y = {}
    for c in range(numCells):
      for m in range(numMutations):
        if I[c, m] == 0:
          Y[c, m] = model.addVar(0, 1, obj=1, vtype=varType, name='Y({0},{1})'.format(c, m))
        elif I[c, m] == 1:
          Y[c, m] = 1

    B = {}
    for p in range(numMutations):
      for q in range(numMutations):
        B[p, q, 1, 1] = model.addVar(0, 1, vtype=varType, obj=0,
                                     name='B[{0},{1},1,1]'.format(p, q))
        B[p, q, 1, 0] = model.addVar(0, 1, vtype=varType, obj=0,
                                     name='B[{0},{1},1,0]'.format(p, q))
        B[p, q, 0, 1] = model.addVar(0, 1, vtype=varType, obj=0,
                                     name='B[{0},{1},0,1]'.format(p, q))
    # model.update()

    for i in range(numCells):
      for p in range(numMutations):
        for q in range(numMutations):
          model.addConstr(Y[i, p] + Y[i, q] - B[p, q, 1, 1] <= 1)
          model.addConstr(-Y[i, p] + Y[i, q] - B[p, q, 0, 1] <= 0)
          model.addConstr(Y[i, p] - Y[i, q] - B[p, q, 1, 0] <= 0)

    for p in range(numMutations):
      for q in range(numMutations):
        model.addConstr(B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2)

    model.Params.ModelSense = GRB.MINIMIZE
    # model.update()
    return model, Y


class StaticILPBounding(BoundingAlgAbstract):
  def __init__(self, ratio = None):
    self.ratio = ratio
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

  n, m = 15, 15
  x = np.random.randint(2, size=(n, m))
  delta = sp.lil_matrix((n, m ))


  staticLPBounding = StaticLPBounding()
  staticLPBounding.reset(x)


  algo = SemiDynamicLPBounding(ratio=None, continuous=True, nThreads=1, tool="Gurobi", prioritySign=-1)
  resetTime = time.time()
  algo.reset(x)
  resetTime = time.time() - resetTime
  print(resetTime)

  xp = np.asarray(x + delta)
  optim = myPhISCS_I(xp)

  print("Optimal answer:", optim)
  print(StaticLPBounding.LP_brief(xp), algo.getBound(delta))
  print(algo.hasState())
  # algo.model.reset()
  algoPrim = algo.model.copy()

  print(algo.hasState(), algoPrim.hasState())
  print(StaticLPBounding.LP_brief(xp), algo.getBound(delta))

  for t in range(5):
    ind = np.nonzero(1 - (x+delta))
    a, b = ind[0][0], ind[1][0]
    delta[a, b] = 1
    print(algo.hasState())
    # algo.model.reset()
    calcTime = time.time()
    bndAdapt = algo.getBound(delta)
    calcTime = time.time() - calcTime
    # print(calcTime)
    algo.getBound(delta)
    bndFull = StaticLPBounding.LP_brief(x+delta) + t + 1
    # print(bndFull == bndAdapt, bndFull, bndAdapt)
    staticLPBoundingBnd = staticLPBounding.getBound(delta)
    # print(bndFull == staticLPBoundingBnd, staticLPBoundingBnd)

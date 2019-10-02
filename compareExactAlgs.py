from Utils.const import *

from interfaces import *
from ErfanFuncs import *
from Boundings.LP import *
from Boundings.MWM import *
from general_BnB import *
# from Boundings.PhISCS_B import *

timeLimit = 50
queue_strategy = "custom"

def solveWith(name, bounding, x):
  ans = copy.copy(x)
  retDict = dict()
  argsToPass = dict()

  if name == "BnB":
    time1 = time.time()
    problem1 = BnB(x, bounding, False)
    solver = pybnb.solver.Solver()
    results1 = solver.solve(problem1,  queue_strategy = queue_strategy, log = None, time_limit = timeLimit)
    retDict["runtime"] = time.time() - time1
    # print(results1.solution_status, results1.termination_condition, results1.objective, results1.nodes, results1.wall_time)
    if results1.solution_status != "unknown":
      delta = results1.best_node.state[0]
      ans = ans + delta
    retDict["nf"] = results1.objective
    retDict["terminationCond"] = results1.termination_condition
    retDict["nNodes"] = str(results1.nodes)
    retDict["internalTime"] = results1.wall_time
    retDict["avgNodeTime"] = retDict["internalTime"] / results1.nodes
  elif callable(name):
    argsNeeded = inspect.getfullargspec(name).args
    for arg in argsNeeded:
      if arg in ['I', 'matrix']:
        argsToPass[arg] = x
      elif arg == 'beta':
        argsToPass[arg] = 0.98
      elif arg == 'alpha':
        argsToPass[arg] = 0.00001
      elif arg == 'csp_solver_path':
        argsToPass[arg] = openwbo_path

    runTime = time.time()
    ans = name(**argsToPass)
    retDict["runtime"] = time.time() - runTime
    if name.__name__ in ["PhISCS_B_external", "PhISCS_I", "PhISCS_B"]:
      retDict["internalTime"] = ans[-1]
      ans = ans[0]
    retDict["nf"] = len(np.where(ans != x)[0])
  else:
    print(f"Method {name} does not exist.")
  return ans, retDict




if __name__ == '__main__':
  scriptName = os.path.basename(__file__).split(".")[0]
  print(f"{scriptName} starts here")
  methods = [
    # (PhISCS_B_external, None),
    (PhISCS_I, None),
    # (PhISCS_B, None),
    # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True)),
    # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = False)),
    # ("BnB", StaticLPBounding(ratio = None, continuous = False)),
    # ("BnB", RandomPartitioning(ascendingOrder=True)),
    # ("BnB", RandomPartitioning(ascendingOrder=False)),
    # ("BnB", DynamicMWMBounding(ascendingOrder=True)),
    # ("BnB", DynamicMWMBounding(ascendingOrder=False)),
    # ("BnB", StaticMWMBounding(ascendingOrder=True)),
    # ("BnB", StaticMWMBounding(ascendingOrder=False)),
    # ("BnB", NaiveBounding()),
    # StaticPhISCSBBounding(splitInto = 2),
    # StaticPhISCSBBounding(splitInto = 3),

  ]
  df = pd.DataFrame(columns=["hash", "n", "m", "nf", "method", "runtime",])
  # n: number of Cells
  # m: number of Mutations
  iterList = itertools.product([ 7, 8, 9, 10 ], # n
                               [ 7, 8, 9, 10 ], # m
                               list(range(3)), # i
                               list(range(len(methods)))
                               )
  iterList = list(iterList)

  # for n, m, i in tqdm(iterList):
  for n, m, i, methodInd in tqdm(iterList):
    # print(n, m, i, methodInd)
    if methodInd == 0: # make new Input
      x = np.random.randint(2, size=(n, m))
      xhash = hash(x.tostring())
      # print(repr(x))
    method, bounding = methods[methodInd]
    # print(bounding.getName())
    ans, info = solveWith(method, bounding, x)
    methodName = method if isinstance(method, str) else method.__name__
    row = {
      "n": str(n),
      "m": str(m),
      "hash": xhash,
      "method": f"{methodName}_{'' if bounding is None else bounding.getName() }",
    }
    row.update(info)
    # print(row)
    df = df.append(row, ignore_index=True)
  print(df)
  nowTime = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
  csvFileName = f"report_{scriptName}_{df.shape}_{nowTime}.csv"
  csvPath = os.path.join(output_folder_path, csvFileName)
  df.to_csv(csvPath)
  print(f"CSV file stored at {csvPath}")





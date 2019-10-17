from Utils.const import *

from Utils.ErfanFuncs import *
from Utils.util import *
import operator
from Utils.interfaces import *

from Boundings.LP import *
from Boundings.MWM import *
from Boundings.CSP import *
from Boundings.Hybrid import *


def fromInterfaceToMethod(boundingAlg):
  def run_func(x):
    boundingAlg.reset(x)
    return boundingAlg.get_bound(sp.lil_matrix(x.shape, dtype = np.int8))
  run_func.__name__ = boundingAlg.get_name() # include arguments in the name
  return run_func


if __name__ == '__main__':
  scriptName = os.path.basename(__file__).split(".")[0]
  print(f"{scriptName} starts here")
  methods = [
    # RandomPartitioning(ascendingOrder=False),
    # RandomPartitioning(ascendingOrder=True),
    # NaiveBounding(),
    # StaticLPBounding(ratio = None, continuous = False),
    SemiDynamicLPBounding(),
    StaticLPBounding(),
    HybridBounding(firstBounding=SemiDynamicLPBounding(), secondBounding=SemiDynamicLPBounding(), ratioNFlips=5),
    # SemiDynamicLPBounding(ratio = None, continuous = True),
    # DynamicMWMBounding(),
    # # StaticMWMBounding(),
    # StaticCSPBounding(splitInto = 2),
    # StaticCSPBounding(splitInto = 3),
  ]
  dfInd = pd.DataFrame(columns=["index", "n",	"m", "nf",	"method", "runtime"])
  missingCols = ["meanUpdateTime", "sdUpdateTime", "medianUpdateTime", "mxUpdateTime", "mnUpdateTime"]
  dfTotal = pd.DataFrame(columns=["method", "resetTime", ] + missingCols)
  n = 12 # n: number of Cells
  m = 12 # m: number of Mutations
  k = 3  # k: number extra edits to introduce
  x = np.random.randint(2, size=(n, m))
  delta = sp.lil_matrix((n, m))
  methodNames = []
  for method in methods:
    methodNames.append(method.get_name())
    resetTime = time.time()
    method.reset(x)
    resetTime = time.time() - resetTime
    # print("68:", method.hasState())
    row = {
      "method": f"{method.get_name()}",
      "resetTime": str(resetTime),
    }
    dfTotal = dfTotal.append(row, ignore_index=True)


  for index in tqdm(range(k)):
    #######  Change one random coordinate ######
    nnzind = np.nonzero(1 - (x + delta))
    a, b = nnzind[0][0], nnzind[1][0]
    delta[a, b] = 1
    ############################################
    for method in methods:
      # print("83:", method.hasState())
      runTime = time.time()
      nf = method.get_bound(delta)
      runTime = time.time() - runTime
      # print("87:", method.hasState())
      row = {
        "index": str(index),
        "n": str(n),
        "m": str(m),
        "method": method.get_name(),
        "runtime": runTime,
        "nf": str(nf),
      }
      dfInd = dfInd.append(row, ignore_index=True)
      # print(row)
  print(dfInd)
  exit(0)
  for methodName in methodNames:
    times = dfInd.loc[dfInd["method"] == methodName]["runtime"].to_numpy()
    dfTotal.loc[dfTotal["method"] == methodName, missingCols] =\
      np.mean(times), np.std(times, ddof = 1), np.median(times), np.min(times), np.max(times)

  nowTime = time.strftime("%m-%d-%H-%M-%S", time.gmtime())

  csvFileName = f"reportTotal_{scriptName}_{dfTotal.shape}_{nowTime}.csv"
  csvPath = os.path.join(output_folder_path, csvFileName)
  dfTotal.to_csv(csvPath)
  print(f"CSV file stored at {csvPath}")


  csvFileName = f"reportIndividual_{scriptName}_{dfInd.shape}_{nowTime}.csv"
  csvPath = os.path.join(output_folder_path, csvFileName)
  dfInd.to_csv(csvPath)
  print(f"CSV file stored at {csvPath}")

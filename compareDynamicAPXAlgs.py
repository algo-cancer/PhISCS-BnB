from Utils.const import *

from ErfanFuncs import *
from util import *
import operator
from interfaces import *

from Boundings.LP import *
from Boundings.MWM import *
from Boundings.PhISCS_B import *

def rename(newname):
  def decorator(f):
    f.__name__ = newname
    return f
  return decorator


def getKPartitionedPhISCS(k):
  @rename(f'partitionedPhISCS_{k}')
  def partitionedPhISCS(x):
    ans = 0
    for i in range(x.shape[1]//k):
      ans += myPhISCS_I(x[:, i * k: (i+1) * k])
    if x.shape[1] % k >= 2:
      ans += myPhISCS_I(x[:, ((x.shape[1]//k) * k): ])
    return ans
  return partitionedPhISCS


def fromInterfaceToMethod(boundingAlg):
  def run_func(x):
    boundingAlg.reset(x)
    return boundingAlg.getBound(sp.lil_matrix(x.shape, dtype = np.int8))
  run_func.__name__ = boundingAlg.getName() # include arguments in the name
  return run_func


if __name__ == '__main__':
  scriptName = os.path.basename(__file__).split(".")[0]
  print(f"{scriptName} starts here")
  methods = [
    NaiveBounding(),
    StaticLPBounding(ratio = None, continuous = False),
    SemiDynamicLPBounding(ratio = None, continuous = False),
    SemiDynamicLPBounding(ratio = None, continuous = True),
    DynamicMWMBounding(),
    StaticMWMBounding(),
    StaticPhISCSBBounding(splitInto = 2),
    StaticPhISCSBBounding(splitInto = 3),
  ]
  dfInd = pd.DataFrame(columns=["index", "n",	"m", "nf",	"method", "runtime"])
  missingCols = ["meanUpdateTime", "sdUpdateTime", "medianUpdateTime", "mxUpdateTime", "mnUpdateTime"]
  dfTotal = pd.DataFrame(columns=["method", "resetTime", ] + missingCols)
  n = 12 # n: number of Cells
  m = 12 # m: number of Mutations
  k = 10  # k: number extra edits to introduce
  x = np.random.randint(2, size=(n, m))
  delta = sp.lil_matrix((n, m))
  methodNames = []
  for method in methods:
    methodNames.append(method.getName())
    resetTime = time.time()
    method.reset(x)
    resetTime = time.time() - resetTime
    row = {
      "method": f"{method.getName()}",
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
      runTime = time.time()
      nf = method.getBound(delta)
      runTime = time.time() - runTime
      row = {
        "index": str(index),
        "n": str(n),
        "m": str(m),
        "method": method.getName(),
        "runtime": runTime,
        "nf": str(nf),
      }
      dfInd = dfInd.append(row, ignore_index=True)
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

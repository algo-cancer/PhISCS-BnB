from Utils.const import *

from ErfanFuncs import *
from util import *
# from collections import defaultdict
# import time
# import pandas as pd
# from tqdm import tqdm
# import itertools, os
# from BnB import *
# from boundingAlgs import *
# from lp_bounding import LP_Bounding, LP_Bounding_direct, LP_Bounding_direct_4
from interfaces import *

from Boundings.LP import *
from Boundings.MWM import *

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
    # myPhISCS_I,
    # myPhISCS_B,
    fromInterfaceToMethod(RandomPartitioning(ascendingOrder = False)),
    fromInterfaceToMethod(RandomPartitioning(ascendingOrder = True)),
    # fromInterfaceToMethod(StaticLPBounding(ratio = 0.5, continuous  = True)),
    # fromInterfaceToMethod(StaticLPBounding(ratio = None, continuous  = False)),
    # fromInterfaceToMethod(StaticLPBounding()),
    # fromInterfaceToMethod(StaticILPBounding()),
    # fromInterfaceToMethod(NaiveBounding()),
    # fromInterfaceToMethod(SemiDynamicLPBounding(ratio = None, continuous  = False)),
    # fromInterfaceToMethod(SemiDynamicLPBounding(ratio = 0.7, continuous  = False)),
    fromInterfaceToMethod(SemiDynamicLPBounding(ratio = None, continuous  = True)),
    # fromInterfaceToMethod(SemiDynamicLPBounding(ratio = 0.7, continuous  = True)),
    fromInterfaceToMethod(DynamicMWMBounding()),
    fromInterfaceToMethod(StaticMWMBounding()),
    # getKPartitionedPhISCS(2),
    # getKPartitionedPhISCS(3),
    # getKPartitionedPhISCS(4),
    # getKPartitionedPhISCS(5),
    # randomPartitionBounding,
    # greedyViolationsPartitionBounding,
    # greedyPartitionBounding,
    # mxWeightedMatchingPartitionBounding,

  ]
  df = pd.DataFrame(columns=["hash", "n",	"m", "nf",	"method", "runtime"])
  # n: number of Cells
  # m: number of Mutations
  iterList = itertools.product([14, 10], # n
                               [14, 10], # m
                               list(range(1)) # i
                               )
  iterList = list(iterList)
  for n, m, ind in tqdm(iterList):
    x = np.random.randint(2, size=(n, m))
    for method in methods:
      runTime = time.time()
      nf = method(x)
      runTime = time.time() - runTime
      row = {
        "n": str(n),
        "m": str(m),
        "hash": hash(x.tostring()),
        "method": f"{method.__name__ }",
        "runtime": str(runTime)[:6],
        "nf": str(nf),
        # "desc": desc
      }
      # print(row)
      df = df.append(row, ignore_index=True)
  print(df)
  nowTime = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
  csvFileName = f"report_{scriptName}_{df.shape}_{nowTime}.csv"
  csvPath = os.path.join(output_folder_path, csvFileName)
  df.to_csv(csvPath)
  print(f"CSV file stored at {csvPath}")



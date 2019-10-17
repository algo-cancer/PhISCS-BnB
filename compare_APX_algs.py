from Utils.const import *
from Utils.util import *
from Utils.interfaces import *

from Boundings.LP import *
from Boundings.MWM import *


if __name__ == '__main__':
  script_name = os.path.basename(__file__).split(".")[0]
  print(f"{script_name} starts here")
  methods = [
    myPhISCS_I,
    myPhISCS_B,
    from_interface_to_method(RandomPartitioning(ascending_order=False)),
    from_interface_to_method(RandomPartitioning(ascending_order=True)),
    from_interface_to_method(SemiDynamicLPBounding(ratio=None, continuous=True, priority_sign=-1,
                                                   change_bound_method=True, for_loop_constrs=True)),
    from_interface_to_method(SemiDynamicLPBounding(ratio=None, continuous=True, priority_sign=-1,
                                                   change_bound_method=False, for_loop_constrs=True)),
    from_interface_to_method(DynamicMWMBounding()),
    from_interface_to_method(StaticMWMBounding()),
  ]
  df = pd.DataFrame(columns=["hash", "n",	"m", "n_flips",	"method", "runtime"])
  # n: number of Cells
  # m: number of Mutations
  iterList = itertools.product([13, ],  # n
                               [13, ],  # m
                               list(range(3))  # i
                               )
  iterList = list(iterList)
  for n, m, ind in tqdm(iterList):
    x = np.random.randint(2, size=(n, m))
    for method in methods:
      runtime = time.time()
      nf = method(x)
      runtime = time.time() - runtime
      row = {
        "n": str(n),
        "m": str(m),
        "hash": get_matrix_hash(x),
        "method": f"{method.__name__ }",
        "runtime": str(runtime)[:6],
        "n_flips": str(nf),
        # "desc": desc
      }
      # print(row)
      df = df.append(row, ignore_index=True)
  print(df)
  now_time = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
  csv_file_name = f"report_{script_name}_{df.shape}_{now_time}.csv"
  csv_path = os.path.join(output_folder_path, csv_file_name)
  df.to_csv(csv_path)
  print(f"CSV file stored at {csv_path}")



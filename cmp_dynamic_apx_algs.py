from Utils.const import *
from Utils.util import *
from Utils.interfaces import *
import operator
from Boundings.LP import *
from Boundings.MWM import *
from Boundings.CSP import *
from Boundings.Hybrid import *


if __name__ == "__main__":
    scriptName = os.path.basename(__file__).split(".")[0]
    print(f"{scriptName} starts here")
    methods = [
        RandomPartitioning(ascending_order=False),
        RandomPartitioning(ascending_order=True),
        SemiDynamicLPBounding(
            ratio=None, continuous=True, priority_sign=-1, change_bound_method=True, for_loop_constrs=True
        ),
        SemiDynamicLPBounding(
            ratio=None, continuous=True, priority_sign=-1, change_bound_method=False, for_loop_constrs=True
        ),
        SemiDynamicLPBounding(
            ratio=None, continuous=True, priority_sign=-1, change_bound_method=False, for_loop_constrs=False
        ),
        DynamicMWMBounding(),
        StaticMWMBounding(),
    ]
    df_individual = pd.DataFrame(columns=["index", "n", "m", "n_flips", "method", "runtime"])
    missing_cols = ["meanUpdateTime", "sd_update_time", "median_update_time", "mx_update_time", "mn_update_time"]
    df_total = pd.DataFrame(columns=["method", "reset_time"] + missing_cols)
    n = 15  # n: number of Cells
    m = 15  # m: number of Mutations
    k = 3  # k: number extra edits to introduce
    x = np.random.randint(2, size=(n, m))
    delta = sp.lil_matrix((n, m))
    method_names = []
    for method in methods:
        method_names.append(method.get_name())
        method_names.append(method.get_name())
        reset_time = time.time()
        method.reset(x)
        reset_time = time.time() - reset_time
        row = {"method": f"{method.get_name()}", "reset_time": str(reset_time)}
        df_total = df_total.append(row, ignore_index=True)

    for index in tqdm(range(k)):
        #######  Change one random coordinate ######
        nnz_ind = np.nonzero(1 - (x + delta))
        a, b = nnz_ind[0][0], nnz_ind[1][0]
        delta[a, b] = 1
        ############################################
        for method in methods:
            reset_time = time.time()
            nf = method.get_bound(delta)
            reset_time = time.time() - reset_time
            row = {
                "index": str(index),
                "n": str(n),
                "m": str(m),
                "method": method.get_name(),
                "runtime": reset_time,
                "n_flips": str(nf),
            }
            df_individual = df_individual.append(row, ignore_index=True)
    print(df_individual)
    for method_name in method_names:
        times = df_individual.loc[df_individual["method"] == method_name]["runtime"].to_numpy()
        df_total.loc[df_total["method"] == method_name, missing_cols] = (
            np.mean(times),
            np.std(times, ddof=1),
            np.median(times),
            np.min(times),
            np.max(times),
        )

    now_time = time.strftime("%m-%d-%H-%M-%S", time.gmtime())

    csv_file_name = f"reportTotal_{scriptName}_{df_total.shape}_{now_time}.csv"
    csv_path = os.path.join(output_folder_path, csv_file_name)
    df_total.to_csv(csv_path)
    print(f"CSV file stored at {csv_path}")

    csv_file_name = f"reportIndividual_{scriptName}_{df_individual.shape}_{now_time}.csv"
    csv_path = os.path.join(output_folder_path, csv_file_name)
    df_individual.to_csv(csv_path)
    print(f"CSV file stored at {csv_path}")

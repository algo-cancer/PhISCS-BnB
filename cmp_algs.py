print("Imports started!")
from Utils.const import *
from Utils.interfaces import *
from Utils.util import *
from Utils.instances import *
from Boundings.LP import *
from Boundings.MWM import *
from Boundings.CSP import *
from Boundings.Hybrid import *
from general_BnB import *
from argparse import ArgumentParser

np.set_printoptions(threshold=sys.maxsize)

try:
    from input import *
except ModuleNotFoundError as e:
    print("Input in Error ----------------")
    print(e)
    print("--------------------------------")
    methods = [
        (PhISCS_I, None),
        (PhISCS_B, None),
        (
            "BnB",
            SemiDynamicLPBounding(
                ratio=None, continuous=True, tool="Gurobi", priority_sign=-1, change_bound_method=True
            ),
        ),
    ]


assert __name__ == "__main__"


parser = ArgumentParser()
parser.add_argument("-n", dest="n", type=int, default=None)
parser.add_argument("-m", dest="m", type=int, default=None)
parser.add_argument("-i", dest="i", type=int, default=None)
parser.add_argument("-t", "--time_limit", dest="time_limit", type=float, default=60)
parser.add_argument("--source_type", dest="source_type", type=int, default=3)
# if source_type in 1, 3
parser.add_argument("-k", dest="k", type=float, default=None)
# if source_type = 3
parser.add_argument("-s", dest="s", type=int, default=None)
# if source_type = 2
parser.add_argument("--instance_index", type=int, default=None)
parser.add_argument("--instance_name", type=str, default=None)
# if n or m is None:
parser.add_argument("--input_config", dest="input_config", type=str, default=None)

# reporting/logging settings
parser.add_argument("--print_rows", action="store_true", default=False)
parser.add_argument("--print_results", action="store_true", default=False)
parser.add_argument("--print_matrix", action="store_true", default=False)
parser.add_argument("--save_results", action="store_true", default=False)
parser.add_argument("--save_solutions", action="store_true", default=False)
args = parser.parse_args()

if args.input_config is None:
    methods = methods # directly from input.py
    k_list = None
else:
    methods, n_list, m_list, k_list, i_number = input_dict[args.input_config]
    n_list, m_list, k_list = list(n_list), list(m_list), list(k_list)

if args.n is not None:
    n_list = [args.n]

if args.m is not None:
    m_list = [args.m]

if args.k is not None:
    k_list = [args.k]

if args.i is not None:
    i_number = args.i


assert n_list is not None and m_list is not None  and i_number is not None

print(f"{len(methods)} number of methods are chosen.")

#########
queue_strategy = "custom"
source_type = ["RND", "MS", "FIXED", "SALEM"][args.source_type]
noisy_list = None
if source_type == "FIXED":
    noisy_list = []
    if args.instance_index is not None:
        noisy = instances[args.instance_index]
    elif args.instance_name is not None:
        if args.instance_name == "list":
            pass # use file_names_list variable from input.py
        else:
            file_names_list = [args.instance_name]
    i_number = len(file_names_list)
print(file_names_list)
def solve_with(name, bounding_algorithm, input_matrix):
    returned_matrix = copy.copy(input_matrix)
    ret_dict = dict()
    args_to_pass = dict()

    if isinstance(name, str) and "BnB" in name:
        time1 = time.time()
        checkBounding = False
        version = 0
        if "True" in name:
            checkBounding = True
        if "1" in name:
            version = 1
        problem1 = BnB(input_matrix, bounding_algorithm, checkBounding, version)
        solver = pybnb.solver.Solver()
        results1 = solver.solve(problem1, queue_strategy=queue_strategy, log=None, time_limit=args.time_limit)
        ret_dict["runtime"] = time.time() - time1
        if results1.solution_status != "unknown":
            returned_delta = results1.best_node.state[0]
            returned_matrix = returned_matrix + returned_delta
        ret_dict["n_flips"] = results1.objective
        ret_dict["termination_condition"] = results1.termination_condition
        ret_dict["n_nodes"] = str(results1.nodes)
        ret_dict["internal_time"] = results1.wall_time
        ret_dict["avg_node_time"] = ret_dict["internal_time"] / results1.nodes
        if bounding_algorithm is not None and hasattr(bounding_algorithm, "times"):
            ret_dict.update(bounding_algorithm.get_times())

    elif callable(name):
        argsNeeded = inspect.getfullargspec(name).args
        for arg in argsNeeded:
            if arg in ["I", "matrix"]:
                args_to_pass[arg] = input_matrix
            elif arg == "beta":
                args_to_pass[arg] = 0.90
            elif arg == "alpha":
                args_to_pass[arg] = 0.00001
            elif arg == "csp_solver_path":
                args_to_pass[arg] = openwbo_path
            elif arg == "time_limit":
                args_to_pass[arg] = args.time_limit

        run_time = time.time()
        returned_output = name(**args_to_pass)
        ret_dict["runtime"] = time.time() - run_time
        if name.__name__ in ["PhISCS_B_external", "PhISCS_I", "PhISCS_B", "PhISCS_B_timed",  "PhISCS_B_2_sat_timed"]:
            ret_dict["internal_time"] = returned_output[-1]
            returned_matrix = returned_output[0]
        if name.__name__ in ["PhISCS_I", "PhISCS_B_external", "PhISCS_B_timed", "PhISCS_B_2_sat_timed"]:
            ret_dict["termination_condition"] = returned_output[-2]
        ret_dict["n_flips"] = len(np.where(returned_matrix != input_matrix)[0])
    else:
        print(f"Method {name} does not exist.")
    return returned_matrix, ret_dict


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    print(f"{script_name} starts here")
    print(args)
    df = pd.DataFrame(columns=["hash", "n", "m", "k", "t", "n_flips", "method", "runtime"])
    # n: number of Cells
    # m: number of Mutations

    if k_list is None or source_type == "RND":
        assert source_type == "RND"
        k_list = [None]

    if source_type == "FIXED":
        n_list, m_list, k_list = [None], [None], [None],
    i_list = list(range(i_number))
    iter_list = itertools.product(n_list, m_list, k_list, i_list, list(range(len(methods))))
    iter_list = list(iter_list)
    print(f"len(iter_list) = {len(iter_list)}")
    x, y, x_hash = None, None, None
    for n, m, k, i, methodInd in tqdm(iter_list):
        if m is None:
            m = n
        if source_type == "RND":
            k = None
        if methodInd == 0:  # make new Input
            if source_type == "RND":
                x = np.random.randint(2, size=(n, m))
            elif source_type == "MS":
                result = get_data_by_ms(
                    n=n, m=m, seed=int(100 * time.time()) % 10000, fn=0, fp=0, na=0, ms_path=ms_path
                )
                ground, noisy, (countFN, countFP, countNA) = result
                k = min(k, np.count_nonzero(ground) - 1)
                x = make_noisy_by_k(ground, int(k))
            elif source_type == "FIXED":
                x = read_matrix_from_file(file_names_list[i])

            elif source_type == "SALEM":
                file_name = f"simNo_{i+1}-s_{args.s}-m_{m}-n_{n}.SC.ground"
                file = simulation_folder_path + file_name
                df_sim = pd.read_csv(file, delimiter="\t", index_col=0)
                # df_sim = df_sim.iloc[: args.n, : args.m]
                y = df_sim.values
                indices_n, indices_m = np.where(y == 1)
                if int(k) > len(indices_n):
                    x = None
                    continue
                else:
                    print(k, len(indices_n), i)
                    x = make_noisy_by_k(y, int(k))
            else:
                raise NotImplementedError("The method not implemented")
            x_hash = get_matrix_hash(x)
            if args.print_matrix:
                print("-" * 20)
                print("Input matrix:")
                print(repr(x))
                print("Matrix Hash:", x_hash)
                print("-" * 20)
        if x is None:
            continue
        else:
            n, m = x.shape
        method, bounding = methods[methodInd]
        method_name = method if isinstance(method, str) else method.__name__
        try:
            # print("solving", method, bounding)
            ans, info = solve_with(method, bounding, x)
            bounding_name = None
            if bounding is None:
                bounding_name = ""
            elif hasattr(bounding, "get_name"):
                bounding_name = bounding.get_name()
            elif hasattr(bounding, "__name__"):
                bounding_name = bounding.__name__
            else:
                bounding_name = "NoNameBounding"

            row = {
                "n": str(n),
                "m": str(m),
                "k": str(k),
                "hash": str(int(x_hash)),
                "method": f"{method_name}_{bounding_name}",
                "cf": is_conflict_free_gusfield_and_get_two_columns_in_coflicts(ans)[0],
                "num_zeros_noisy": str(int(np.count_nonzero(1-x))),
                "num_ones_noisy": str(int(np.count_nonzero(x))),
                "t": str(args.time_limit),
            }
            if args.save_solutions:
                print(type(ans))
                if isinstance(ans, np.ndarray):
                    ans_file_name = f"solution_{row['hash']}_{row['method']}"
                    np.savetxt(ans_file_name, ans, fmt = "%d")
                    print(f"numpy file stored at {ans_file_name}")
                else:
                    print("No answer!")
            if source_type == "SALEM":
                row["s"] = str(args.s)
                row["file_name"] = file_name
                row["simNo"] = str(i+1)
                row["num_zeros_ground"] = str(int(np.count_nonzero(1 - y))),
                row["num_ones_ground"] =  str(int(np.count_nonzero(y))),
            elif source_type == "FIXED":
                row["file_name"] = file_names_list[i]
            row.update(info)
            if args.print_rows:
                print(row)
            df = df.append(row, ignore_index=True)
        except Exception as e:
            print("********** Error {{{{{{{{{{")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print(e)
            print(repr(x))
            print(method_name)
            print("}}}}}}}}}} Error **********")
    if args.print_results:
        summary_columns = [
            "method",
            "cf",
            "n_flips",
            "runtime",
            "n_nodes",
            "model_preparation_time",
            "optimization_time",
        ]
        summary_columns = (column for column in summary_columns if column in df.columns)
        print(df[summary_columns])
        # print(">>>", df.loc[0, "n_flips"], df.loc[1, "n_flips"], df.loc[0, "runtime"], df.loc[1, "optimization_time"])
    if args.save_results:
        now_time = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
        if source_type == "SALEM":
            csv_file_name = f"n_{args.n}-m_{args.m}-s_{args.s}-k_{int(args.k)}.csv"
        else:
            csv_file_name = f"{script_name}_{args.n},{args.m},{len(methods)}_{now_time}.csv"

        csv_path = os.path.join(output_folder_path, csv_file_name)
        df.to_csv(csv_path)
        print(f"CSV file stored at {csv_path}")

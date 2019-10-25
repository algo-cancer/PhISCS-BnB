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
    from input import methods
except ModuleNotFoundError as e:
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
finally:
    print(f"{len(methods)} number of methods are chosen.")

assert __name__ == "__main__"


parser = ArgumentParser()
parser.add_argument("-n", dest="n", type=int, default=None)
parser.add_argument("-m", dest="m", type=int, default=None)
parser.add_argument("-s", dest="s", type=int, default=None)
parser.add_argument("-i", dest="i", type=int, default=1)
parser.add_argument("-k", dest="k", type=float, default=None)
parser.add_argument("--instance_index", type=int, default=0)
parser.add_argument("--print_rows", action="store_true", default=False)
parser.add_argument("--print_results", action="store_true", default=False)
parser.add_argument("--save_results", action="store_true", default=False)
parser.add_argument("--source_type", dest="source_type", type=int, default=3)
parser.add_argument("-t", "--time_limit", dest="time_limit", type=float, default=60)
args = parser.parse_args()


#########
queue_strategy = "custom"
source_type = ["RND", "MS", "FIXED", "SALEM"][args.source_type]
noisy = instances[args.instance_index]


def solve_with(name, bounding_algorithm, input_matrix):
    returned_matrix = copy.copy(input_matrix)
    ret_dict = dict()
    args_to_pass = dict()

    if name == "BnB":
        time1 = time.time()
        problem1 = BnB(input_matrix, bounding_algorithm, False)
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
            ret_dict.update(bounding_algorithm.times)
    elif name == "OldBnB":
        time1 = time.time()
        problem1 = Phylogeny_BnB(input_matrix, bounding_algorithm, bounding_algorithm.__name__)
        solver = pybnb.solver.Solver()
        results1 = solver.solve(problem1, queue_strategy=queue_strategy, log=None, time_limit=args.time_limit)
        ret_dict["runtime"] = time.time() - time1
        if results1.solution_status != "unknown":
            flip_list = results1.best_node.state[0]
            assert np.all(returned_matrix[tuple(np.array(flip_list).T)] == 0)
            returned_matrix[tuple(np.array(flip_list).T)] = 1
        ret_dict["n_flips"] = results1.objective
        ret_dict["termination_cond"] = results1.termination_condition
        ret_dict["n_nodes"] = str(results1.nodes)
        ret_dict["internal_time"] = results1.wall_time
        ret_dict["avg_node_time"] = ret_dict["internal_time"] / results1.nodes

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
        returned_matrix = name(**args_to_pass)
        ret_dict["runtime"] = time.time() - run_time
        if name.__name__ in ["PhISCS_B_external", "PhISCS_I", "PhISCS_B"]:
            ret_dict["internal_time"] = returned_matrix[-1]
            returned_matrix = returned_matrix[0]
        ret_dict["n_flips"] = len(np.where(returned_matrix != input_matrix)[0])
    else:
        print(f"Method {name} does not exist.")
    return returned_matrix, ret_dict


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    print(f"{script_name} starts here")
    print(args)
    df = pd.DataFrame(columns=["hash", "n", "m", "k", "n_flips", "method", "runtime"])
    # n: number of Cells
    # m: number of Mutations

    if args.k is None:
        k_list = [0.1]
    else:
        k_list = [args.k]
    if args.n is None or args.m is None:  # if n and m not given use our looping
        # 20, 30 , 40, 50, 60, 70, 80, 90, 40, 80, 100, 120, 160
        iterList = itertools.product(
            range(5, 21), range(5, 21), k_list, list(range(10)), list(range(len(methods)))
        )  # n  # m  # i
    else:
        iterList = itertools.product([args.n], [args.m], k_list, list(range(args.i)), list(range(len(methods))))

    iterList = list(iterList)
    x, x_hash = None, None
    for n, m, k, i, methodInd in tqdm(iterList):
        if m is None:
            m = n
        if source_type == "RND":
            k = None
        if methodInd == 0:  # make new Input
            if source_type == "RND":
                x = np.random.randint(2, size=(n, m))
            elif source_type == "MS":
                ground, noisy, (countFN, countFP, countNA) = get_data_by_ms(
                    n=n, m=m, seed=int(100 * time.time()) % 10000, fn=k, fp=0, na=0, ms_path=ms_path
                )
                x = noisy
            elif source_type == "FIXED":
                x = noisy
            elif source_type == "SALEM":
                file_name = f"simNo_{i+1}-s_{args.s}-m_{args.m}-n_{args.n}.SC.ground"
                file = simulation_folder_path + file_name
                df_sim = pd.read_csv(file, delimiter="\t", index_col=0)
                # df_sim = df_sim.iloc[: args.n, : args.m]
                indices_n, indices_m = np.where(df_sim.values == 1)
                if int(k) > len(indices_n):
                    continue
                x = make_noisy_by_k(df_sim.values, int(k))
            else:
                raise NotImplementedError("The method not implemented")
            x_hash = get_matrix_hash(x)
        if x is None:
            continue
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
                "hash": x_hash,
                "method": f"{method_name}_{bounding_name}",
                "cf": is_conflict_free_gusfield_and_get_two_columns_in_coflicts(ans)[0],
                "num_zeros": str(np.count_nonzero(1-x)),
                "num_ones": str(np.count_nonzero(x)),
            }
            if source_type == "SALEM":
                row["s"] = str(args.s)
                row["file_name"] = file_name
                row["simNo"] = str(i+1)
            row.update(info)
            if args.print_rows:
                print(row)
            df = df.append(row, ignore_index=True)
        except Exception as e:
            print("********** Error {{{{{{{{{{")
            traceback.print_tb()
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
        # csv_file_name = f"{script_name}_{args.n},{args.m},{len(methods)}_{now_time}.csv"
        csv_file_name = f"n_{args.n}-m_{args.m}-s_{args.s}-k_{int(args.k)}.csv"
        csv_path = os.path.join(output_folder_path, csv_file_name)
        df.to_csv(csv_path)
        print(f"CSV file stored at {csv_path}")

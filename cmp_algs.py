from Utils.const import *
from Utils.interfaces import *
from Utils.ErfanFuncs import *
from Utils.util import *
from Boundings.LP import *
from Boundings.MWM import *
from general_BnB import *
from Boundings.CSP import *
from Boundings.Hybrid import *
from argparse import ArgumentParser

assert __name__ == "__main__"


parser = ArgumentParser()
parser.add_argument("-n", dest="n", type=int, default=None)
parser.add_argument("-m", dest="m", type=int, default=None)
parser.add_argument("-i", dest="i", type=int, default=1)
parser.add_argument("-s", "--source_type", dest="source_type", type=int, default=0)
parser.add_argument("-k", dest="k", type=float, default=None)
parser.add_argument("--instance_index", type=int, default=0)
parser.add_argument("--print_rows", action="store_true", default=False)
parser.add_argument("--print_results", action="store_true", default=False)
parser.add_argument("--save_results", action="store_true", default=False)
parser.add_argument("-t", "--time_limit", dest="time_limit", type=float, default=60)
args = parser.parse_args()


#########
queue_strategy = "custom"
source_type = ["RND", "MS", "FIXED"][args.source_type]
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
                args_to_pass[arg] = 0.98
            elif arg == "alpha":
                args_to_pass[arg] = 0.00001
            elif arg == "csp_solver_path":
                args_to_pass[arg] = openwbo_path

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
    methods = [
        # ("BnB", HybridBounding(
        #     firstBounding=SemiDynamicLPBounding(),
        #     secondBounding=DynamicMWMBounding(),
        #     ratioNFlips=5,
        # )),
        # ("BnB", HybridBounding(
        #     firstBounding=SemiDynamicLPBounding(),
        #     secondBounding=DynamicMWMBounding(),
        #     ratioNFlips=10,
        # )),
        # ("BnB", HybridBounding(
        #     firstBounding=SemiDynamicLPBounding(),
        #     secondBounding=DynamicMWMBounding(),
        #     ratioNFlips=15,
        # )),
        # # (PhISCS_B_external, None),
        (PhISCS_I, None),
        (PhISCS_B, None),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True)),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True, tool = "Gurobi", prioritySign = 1)),
        # ("OldBnB", lb_lp_gurobi),
        (
            "BnB",
            SemiDynamicLPBounding(
                ratio=None,
                continuous=True,
                tool="Gurobi",
                priority_sign=-1,
                change_bound_method=True,
            ),
        ),
        # (
        #     "BnB",
        #     SemiDynamicLPBounding(
        #         ratio=None,
        #         continuous=True,
        #         tool="Gurobi",
        #         priority_sign=-1,
        #         change_bound_method=False,
        #         for_loop_constrs=True,
        #     ),
        # ),
        # (
        #     "BnB",
        #     SemiDynamicLPBounding(
        #         ratio=None,
        #         continuous=True,
        #         tool="Gurobi",
        #         priority_sign=-1,
        #         change_bound_method=False,
        #         for_loop_constrs=False,
        #     ),
        # ),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True, tool = "Gurobi", prioritySign = -1)),
        # ("BnB", SemiDynamicLPBoundingBoundChange(ratio=None, continuous = True, tool = "Gurobi", prioritySign = -1)),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous=True, tool="Gurobi", priority_sign=-1)),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True, tool = "Gurobi", prioritySign = -1)),
        # ("OldBnB", lb_lp_gurobi),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True, tool = "Gurobi", prioritySign = -1)),
        # ("OldBnB", lb_lp_gurobi),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True, tool = "Gurobi", prioritySign = 1)),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True, tool = "Gurobi", prioritySign = -1)),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True, tool = "ORTools")),
        # ("OldBnB", lb_lp_ortools),
        # ("BnB", SemiDynamicLPBounding(ratio=0.8, continuous = True)),
        # ("BnB", SemiDynamicLPBounding(ratio=0.7, continuous = True)),
        # ("BnB", SemiDynamicLPBounding(ratio=0.5, continuous = True)),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = False)),
        # ("BnB", StaticLPBounding(ratio = None, continuous = True)),
        # ("BnB", RandomPartitioning(ascendingOrder=True)),
        # ("BnB", RandomPartitioning(ascendingOrder=False)),
        # ("OldBnB", lb_max_weight_matching),
        # ("BnB", SemiDynamicCSPBounding(splitInto=2)),
        ("BnB", DynamicMWMBounding(ascending_order=True)),
        ("BnB", DynamicMWMBounding(ascending_order=False)),
        # ("BnB", StaticMWMBounding(ascending_order=True)),
        # ("BnB", StaticMWMBounding(ascending_order=False)),
        # ("OldBnB", lb_max_weight_matching),
        # ("OldBnB", lb_lp_ortools),
        # ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True)),
        # ("OldBnB", lb_phiscs_b),
        # ("OldBnB", lb_openwbo),
        # ("OldBnB", lb_gurobi),
        # ("OldBnB", lb_greedy),
        # ("OldBnB", lb_random),
        # ("BnB", RandomPartitioning(ascending_order=True)),
        # ("BnB", RandomPartitioning(ascending_order=False)),
        # ("BnB", StaticMWMBounding(ascendingOrder=True)),
        # ("BnB", StaticMWMBounding(ascendingOrder=False)),
        # ("BnB", NaiveBounding()),
        # ("BnB", StaticCSPBounding(splitInto = 2)),
        # ("BnB", StaticCSPBounding(splitInto = 3)),
        # ("BnB", StaticCSPBounding(splitInto = 4)),
        # ("BnB", StaticCSPBounding(splitInto = 5)),
        # ("BnB", HybridBounding(firstBounding=SemiDynamicLPBounding(ratio=None, continuous=True, tool="Gurobi", prioritySign=-1),
        #                        secondBounding=DynamicMWMBounding(ascendingOrder=False),
        #                        ratioNFlips=10)),
    ]
    df = pd.DataFrame(columns=["hash", "n", "m", "n_flips", "method", "runtime"])
    # n: number of Cells
    # m: number of Mutations


    if args.k is None:
        k_list = [0.1, ]
    else:
        k_list = [args.k]
    if args.n is None or args.m is None:  # if n and m not given use our looping
        # 20, 30 , 40, 50, 60, 70, 80, 90, 40, 80, 100, 120, 160
        iterList = itertools.product(range(5, 21), range(5, 21), k_list, list(range(10)), list(range(len(methods))))  # n  # m  # i
    else:
        iterList = itertools.product([args.n], [args.m], k_list, list(range(args.i)), list(range(len(methods))))

    iterList = list(iterList)
    x, x_hash = None, None
    for n, m, k, i, methodInd in tqdm(iterList):
        if m is None:
            m = n
        if methodInd == 0:  # make new Input
            if source_type == "RND":
                x = np.random.randint(2, size=(n, m))
            elif source_type == "MS":
                ground, noisy, (countFN, countFP, countNA) = get_data(
                    n=n, m=m, seed=int(100 * time.time()) % 10000, fn=k, fp=0, na=0, ms_path=ms_path
                )
                x = noisy
            elif source_type == "FIXED":
                x = noisy
            else:
                raise NotImplementedError("The method not implemented")
            x_hash = get_matrix_hash(x)

        method, bounding = methods[methodInd]
        method_name = method if isinstance(method, str) else method.__name__
        try:
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
                "hash": x_hash,
                "method": f"{method_name}_{bounding_name}",
                "cf": is_conflict_free_gusfield_and_get_two_columns_in_coflicts(ans)[0],
            }
            row.update(info)
            if args.print_rows:
                print(row)
            df = df.append(row, ignore_index=True)
        except Exception as e:
            print("********** Error {{{{{{{{{{")
            print(e)
            print(repr(x))
            print(method_name)
            print("}}}}}}}}}} Error **********")
    if args.print_results:
        summary_columns = ["method", "cf", "n_flips", "runtime", "n_nodes"]
        summary_columns = (column for column in summary_columns if column in df.columns)
        print(df[summary_columns])
    if args.save_results:
        now_time = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
        # csvFileName = f"{scriptName}_{nowTime}.csv"
        csv_file_name = f"{script_name}_{args.n},{args.m},{len(methods)}_{now_time}.csv"
        csv_path = os.path.join(output_folder_path, csv_file_name)
        df.to_csv(csv_path)
        print(f"CSV file stored at {csv_path}")

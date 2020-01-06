from algorithms.PhISCS import PhISCS_B, PhISCS_I
from algorithms.BnB import bnb_solve
from boundings.two_sat import TwoSatBounding
from algorithms.twosat import twosat_solver
from utils.util import *

bounding_algs_index = 0
bounding_algs = [
        # TwoSatBounding(heuristic_setting=None, n_levels=1, compact_formulation=False),
        # TwoSatBounding(heuristic_setting=None, n_levels=2, compact_formulation=False),
        # TwoSatBounding(heuristic_setting=None, n_levels=1, compact_formulation=True),
        TwoSatBounding(heuristic_setting=None, n_levels=2, compact_formulation=False), # Real Data
        # TwoSatBounding(heuristic_setting=[True, True, False, True, True], n_levels=2, compact_formulation=False),
        # TwoSatBounding(heuristic_setting=[True, True, False, True, True], n_levels=2, compact_formulation=True),
        TwoSatBounding(heuristic_setting=[True, True, False, True, True], n_levels=1, compact_formulation=True), # Simulation
    ]


def solve_by(func, input_matrix, na_value):
    global bounding_algs_index
    global bounding_algs
    print()
    printf(f"Solving by {func.__name__}")
    required_args = inspect.getfullargspec(func).args
    args_to_pass = dict()
    for arg in required_args:
        if arg in ["I", "matrix"]:
            args_to_pass[arg] = input_matrix
        elif arg == "beta":
            args_to_pass[arg] = 0.90
        elif arg == "alpha":
            args_to_pass[arg] = 0.00001
        elif arg == "csp_solver_path":
            args_to_pass[arg] = openwbo_path
        elif arg == "time_limit":
            if args.time_limit <= 0:
                time_limit = None
            else:
                time_limit = args.time_limit
            args_to_pass[arg] = time_limit
        elif arg == "na_value":
            args_to_pass[arg] = na_value
        elif arg == "bounding_algorithm":
            args_to_pass[arg] = bounding_algs[bounding_algs_index]
        # elif arg ==

    solve_time = time.time()
    result = func(**args_to_pass)
    solve_time = time.time() - solve_time
    matrix_output = result[0]
    flips_0_1, flips_1_0, flips_na_0, flips_na_1 = count_flips(I=input_matrix, sol_Y=matrix_output, na_value=na_value)
    printf(f"#0->1: {flips_0_1}")
    printf(f"#1->0: {flips_1_0}")
    printf(f"#na->0: {flips_na_0}")
    printf(f"#na->1: {flips_na_1}")
    cf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(matrix_output)
    printf(f"CF?: {cf}")
    printf(f"solve_time: {solve_time:.3f}")

    if args.o is not None:
        df_output = pd.DataFrame(matrix_output)
        df_output.columns = df_input.columns
        df_output.index = df_input.index
        df_output.index.name = "cellIDxmutID"
        filename = os.path.splitext(os.path.basename(args.i))[0]
        file = os.path.join(args.o, filename)
        output_filename = f"{file}_{func.__name__}_{matrix_output.shape}.CFMatrix"
        df_output.to_csv(output_filename, sep="\t")
        print("Output file is saved at ", output_filename)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", dest="n", type=int, default=None, help="Number of cells to include")
    parser.add_argument("-m", dest="m", type=int, default=None, help="Number of mutations to include")
    parser.add_argument("-i", dest="i", type=str, default=None, help="Input filename")
    parser.add_argument("-b", dest="b", action='store_true', help="BnB")
    parser.add_argument("-c", dest="c", action='store_true', help="CSP")
    parser.add_argument("-g", dest="g", action='store_true', help="ILP")
    parser.add_argument("-r", dest="r", type=float, default=0.8, help="An argument for Blocking")
    parser.add_argument("-o", dest="o", type=str, default=None, help="Output Folder")
    parser.add_argument("-t", dest="time_limit", type=int, default=0, help="Time limit for each algorithm")
    args = parser.parse_args()

    df_input = pd.read_csv(args.i, delimiter="\t", index_col=0)
    if args.n is not None:
        df_input = df_input[:args.n]
    if args.m is not None:
        df_input = df_input[df_input.columns[:args.m]]
    input_matrix = df_input.values

    df_output = pd.DataFrame()
    na_value = infer_na_value(input_matrix)
    printf(f"Size: {input_matrix.shape}")
    printf(f"NA value: {na_value}")
    printf(f"#Zeros: {len(np.where(input_matrix == 0)[0])}")
    printf(f"#Ones: {len(np.where(input_matrix == 1)[0])}")
    printf(f"#NAs: {len(np.where(input_matrix == na_value)[0])}")

    if args.b:
        # solve_by(twosat_solver, input_matrix, na_value)
        bounding_algs_index = 0
        solve_by(bnb_solve, input_matrix, na_value)
    if args.c:
        solve_by(PhISCS_B, input_matrix, na_value)
    if args.g:
        solve_by(PhISCS_I, input_matrix, na_value)

    # exit(0)
    # For Erfan's use:
    # for bounding_algs_index in range(len(bounding_algs)):
    #     print(bounding_algs[bounding_algs_index].get_name())
    #     solve_by(bnb_solve, input_matrix, na_value)

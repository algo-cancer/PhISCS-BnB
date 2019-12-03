from Utils.const import *
from Utils.util import *
from Utils.interfaces import *

from Boundings.LP import *
from Boundings.LP_APX_b import *
from Boundings.MWM import *
from Boundings.two_sat import *
from Boundings.graph import *


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    print(f"{script_name} starts here")
    methods = [
        # myPhISCS_I,
        from_interface_to_method(DynamicMWMBounding()),
        from_interface_to_method(GraphicBounding(version=7)),
        # from_interface_to_method(GraphicBounding(version=8)),
        # from_interface_to_method(RandomPartitioning(ascending_order=False)),
        from_interface_to_method(two_sat(priority_version=1, formulation_version=0, formulation_threshold=0)),
        # from_interface_to_method(two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.25)),
        # from_interface_to_method(two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.5)),
        # from_interface_to_method(two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.75)),
        # from_interface_to_method(two_sat(priority_version=1, formulation_version=0, formulation_threshold=1)),
        # myPhISCS_B,
        from_interface_to_method(SemiDynamicLPBounding()),
        # from_interface_to_method(SubsampleLPBounding_b(SubsampleLPBounding_b.n_const_func_maker(3, 2000))),
        # from_interface_to_method(SubsampleLPBounding_b(SubsampleLPBounding_b.n_const_func_maker(2, 2000))),
        # from_interface_to_method(SubsampleLPBounding_b(SubsampleLPBounding_b.n_const_func_maker(1.7, 2000))),
        # from_interface_to_method(SubsampleLPBounding_b(SubsampleLPBounding_b.n_const_func_maker(1.1, 2000))),
        # from_interface_to_method(SubsampleLPBounding_b(SubsampleLPBounding_b.n_const_func_maker(1.5, 0))),
        # from_interface_to_method(SubsampleLPBounding_b(SubsampleLPBounding_b.n_const_func_maker(1, 0))),
        # from_interface_to_method(RandomPartitioning(ascending_order=True)),
        # from_interface_to_method(
        #     SemiDynamicLPBounding(
        #         ratio=None, continuous=True, priority_sign=-1, change_bound_method=True, for_loop_constrs=True
        #     )
        # ),
        # from_interface_to_method(
        #     SemiDynamicLPBounding(
        #         ratio=None, continuous=True, priority_sign=-1, change_bound_method=False, for_loop_constrs=True
        #     )
        # ),
        # from_interface_to_method(
        #     SemiDynamicLPBounding(
        #         ratio=None, continuous=True, priority_sign=-1, change_bound_method=False, for_loop_constrs=False
        #     )
        # ),
        # from_interface_to_method(StaticMWMBounding()),
    ]
    df = pd.DataFrame(columns=["hash", "n", "m", "n_flips", "method", "runtime"])
    # n: number of Cells
    # m: number of Mutations
    iterList = itertools.product([17], [17], list(range(5)))  # n  # m  # i
    iterList = list(iterList)
    iterList = [(None, None, None)]
    for n, m, ind in tqdm(iterList):
        # x = np.random.randint(2, size=(n, m))
        if True:
            # x = read_matrix_from_file("simNo_1-s_10-m_20-n_80-k_35.SC.noisy")
            # x = read_matrix_from_file()
            # x = read_matrix_from_file("../noisy_500/simNo_1-s_4-m_500-n_500-fn_0.1-k_5217.SC.noisy")
            xx = np.loadtxt("fivehmatrix.txt")

            # for ss in range(200, 201):
            if True:
                x = xx[:17, :17]
                # print(x)
                n, m = x.shape
                print(n, m)
                print(len(methods))
                ind = 0
                for method in methods:
                    print(method)
                    runtime = time.time()
                    result = method(x)
                    runtime = time.time() - runtime
                    if isinstance(result, tuple):
                        nf = result[0]
                        methodname = result[1]
                    else:
                        nf = result
                        methodname = f"{method.__name__ }"
                    row = {
                        "n": str(n),
                        "m": str(m),
                        "hash": get_matrix_hash(x),
                        "method": methodname,
                        "runtime": str(runtime)[:6],
                        "n_flips": str(nf),
                    }
                    df = df.append(row, ignore_index=True)
                    print(row)
    print(df)
    now_time = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
    csv_file_name = f"report_{script_name}_{df.shape}_{now_time}.csv"
    csv_path = os.path.join(output_folder_path, csv_file_name)
    df.to_csv(csv_path)
    print(f"CSV file stored at {csv_path}")

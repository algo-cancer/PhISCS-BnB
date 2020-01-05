from Utils.const import *
from Utils.util import *
from Utils.interfaces import *

from Boundings.LP import *
from Boundings.LP_APX_b import *


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    print(f"{script_name} starts here")
    methods = [
        from_interface_to_method(SemiDynamicLPBounding()),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 200 * n * m,)),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 9 * (3 * n ** 1.05 + 1) * m ** 0.9,)),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 10 * n * m ** 1.5,)),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 10 * n**1.5 * m ,)),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 100 * n**0.5 * m**1.5 ,)),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 300 * (n + m ** 2),)),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: (3 * n + 1) * m * (m - 1))),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 0.2*(3 * n + 1) * m * (m - 1))),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 0.1*(3 * n + 1) * m * (m - 1))),
        from_interface_to_method(SubsampleLPBounding_b(lambda n, m: 0.01*(3 * n + 1) * m * (m - 1))),
    ]
    df = pd.DataFrame(columns=["hash", "n", "m", "n_flips", "method", "runtime"])
    # n: number of Cells
    # m: number of Mutations
    # [10, 20, 30, 40, 50]
    iterList = itertools.product([10, 20, 30, 40, 50, 60, 70], [10, 20, 30, 40, 50, 60, 70], list(range(5)))  # n  # m  # i
    iterList = list(iterList)
    for n, m, ind in tqdm(iterList):
        x = np.random.randint(2, size=(n, m))
        for method in methods:
            runtime = time.time()
            ret = timed_run(method, {"x":x}, time_limit = 20)

            runtime = time.time() - runtime
            row = {
                "n": str(n),
                "m": str(m),
                "hash": get_matrix_hash(x),
                "runtime": str(runtime),
                "internal_time": str(ret["runtime"]),
                "termination_condition": ret['termination_condition'],
            }
            if ret['termination_condition'] == "success":
                row.update({
                    "method": f'{ret["output"][1] }',
                    "n_flips": str(ret["output"][0]),
                })
            df = df.append(row, ignore_index=True)
    print(df)
    now_time = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
    csv_file_name = f"report_{script_name}_{df.shape}_{now_time}.csv"
    csv_path = os.path.join(output_folder_path, csv_file_name)
    df.to_csv(csv_path)
    print(f"CSV file stored at {csv_path}")

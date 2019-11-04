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
    iterList = itertools.product([20, 40, 80, 160, 320], [20, 40, 80, 160, 320], list(range(3)))  # n  # m  # i
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
            }
            df = df.append(row, ignore_index=True)
    print(df)
    now_time = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
    csv_file_name = f"report_{script_name}_{df.shape}_{now_time}.csv"
    csv_path = os.path.join(output_folder_path, csv_file_name)
    df.to_csv(csv_path)
    print(f"CSV file stored at {csv_path}")

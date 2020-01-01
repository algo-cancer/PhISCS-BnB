from utils.const import *


def all_None(*args):
    return args.count(None) == len(args)


def now():
    return f"[{datetime.datetime.now().strftime('%m/%d %H:%M:%S')}]"


def printf(s):
    print(f"[{now()}] ", end="")
    print(s, flush=True)


def print_line(depth=1, shift=1):
    """A debugging tool!  """
    stack = inspect.stack()
    for i in range(shift, min(len(stack), depth + shift)):
        info = stack[i]
        for j in range(i - 1):
            print("\t", end="")
        print(f"Line {info.lineno} in {info.filename}, Function: {info.function}")


def print_line_iter(depth=1, precision=5):
    """A debugging tool!  """
    last_time = 0
    while(True):
        last_time = time.time() - last_time
        print(round(last_time, precision), end="")
        print_line(depth=depth, shift=2)
        print(flush=True)
        last_time = time.time()
        yield
        # return


def null_prob(n01: int, n11: int, p: float) -> float:
    """
    Probability of observing n01 number of (0, 1)s from n01+n11 number of original intersections,
     if the probability of flipping is p.
    :param n01:
    :param n11:
    :param p:
    :return:
    """
    return 1 - st.binom.cdf(n01-1, n01 + n11, p)


def compute_relation_matrices(matrix, na_value=None, fn_rate=0.1, prob=0.01):
    """
    @param matrix: cells on rows and mutations on columns
    @param na_value: usually 2 or 3
    @param fn_rate: the rate in which 1s became 0s
    @param prob: Acceptable probability of missing an event.
    @return:
    """
    if na_value is not None:
        matrix = matrix.copy()
        matrix[matrix == na_value] = 0.5
    # else:  # use matrix as is
    # matrix: Final = matrix  # no change for matrix in this function from now on

    rates = np.zeros((2, 2, matrix.shape[0], matrix.shape[0]))
    decendent_row = np.zeros((matrix.shape[0], matrix.shape[0]), dtype = np.bool)
    ind = 0
    for row_i in range(rates.shape[2]):
        for row_j in range(rates.shape[3]):
            for a, b in itertools.product(range(2), repeat=2):
                rates[a, b, row_i, row_j] = np.sum(np.logical_and(matrix[row_i] == a, matrix[row_j] == b))

            n_ones = rates[1, 1, row_i, row_j]
            n_switched = rates[0, 1, row_i, row_j]
            null_pobability = null_prob(n_switched, n_ones, fn_rate)
            decendent_row[row_i, row_j] = null_pobability > prob
            # if prob < null_pobability < 1:
            #     # print(row_i, row_j, ind)
            #     # print(n_switched, n_ones, decendent_row[row_i, row_j], null_pobability )
            #     ind += 1

    intersection_col = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.bool)
    for intersection_col_i in range(intersection_col.shape[0]):
        for intersection_col_j in range(intersection_col.shape[1]):
            intersection_col[intersection_col_i, intersection_col_j] = \
                np.sum(np.logical_and(matrix[:, intersection_col_i] == 1, matrix[:, intersection_col_j] == 1))

    return rates, intersection_col, decendent_row


def read_matrix_from_file(
        file_name="simNo_2-s_4-m_50-n_50-k_50.SC.noisy",
        args={"simNo":2, "s": 4, "m": 50, "n":50, "k":50, "kind": "noisy"}, folder_path=noisy_folder_path):
    assert file_name is not None or args is not None, "give an input"

    if file_name is None:
        file_name = f"simNo_{args['simNo']}-s_{args['s']}-m_{args['m']}-n_{args['n']}.SC.{args['kind']}"
        if args['kind'] == "noisy":
            folder_path = noisy_folder_path
        elif args['kind'] == "ground":
            folder_path = simulation_folder_path
    if folder_path is not None:
        file = os.path.join(folder_path, file_name)
    else:
        file = file_name
    df_sim = pd.read_csv(file, delimiter="\t", index_col=0)
    return df_sim.values


def timed_run(func, args, time_limit=1):
    def internal_func(shared_dict):
        args_to_pass = dict()
        args_info = inspect.getfullargspec(func)
        args_needed = args_info.args
        args_defaults = args_info.defaults
        for ind, arg in enumerate(args_needed):
            if arg in shared_dict["input"]:
                args_to_pass[arg] = shared_dict["input"][arg]
            elif len(args_needed) - ind <= len(args_defaults) :
                args_to_pass[arg] = args_defaults[ind - len(args_needed)]
            else:
                assert False, f"value for argument {arg} was not found!"

        runtime = time.time()
        shared_dict["output"] = func(**args_to_pass)
        runtime = time.time() - runtime
        shared_dict["runtime"] = runtime

    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    shared_dict["input"] = args
    p = multiprocessing.Process(target=internal_func, name="internal_func", args=(shared_dict,))
    p.start()
    p.join(time_limit)
    # If thread is active
    if p.is_alive():
        p.terminate()
        # p.join()
        shared_dict["output"] = None
        shared_dict["termination_condition"] = 'time_limit'
        shared_dict["runtime"] = time_limit
    else:
        shared_dict["termination_condition"] = 'success'
    return shared_dict


def get_matrix_hash(x, digits = 7) -> int:
    """
    :param x: Accepts any object but usually just a matrix.
    :return:
    """
    return hash(x.tostring()) % (10**7)


def get_a_coflict(D, p, q):
    # todo: oneone is not important you can get rid of
    oneone = None
    zeroone = None
    onezero = None
    for r in range(D.shape[0]):
        if D[r, p] == 1 and D[r, q] == 1:
            oneone = r
        if D[r, p] == 0 and D[r, q] == 1:
            zeroone = r
        if D[r, p] == 1 and D[r, q] == 0:
            onezero = r
        if oneone != None and zeroone != None and onezero != None:
            return (p, q, oneone, zeroone, onezero)
    return None


def is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I):
    def sort_bin(a):
        b = np.transpose(a)
        b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
        idx = np.argsort(b_view.ravel())[::-1]
        c = b[idx]
        return np.transpose(c), idx

    I = I.copy()
    I[I == 2] = 0
    O, idx = sort_bin(I)
    # TODO: delete duplicate columns
    # print(O, '\n')
    Lij = np.zeros(O.shape, dtype=int)
    for i in range(O.shape[0]):
        maxK = 0
        for j in range(O.shape[1]):
            if O[i, j] == 1:
                Lij[i, j] = maxK
                maxK = j + 1
    # print(Lij, '\n')
    Lj = np.amax(Lij, axis=0)
    # print(Lj, '\n')
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            if O[i, j] == 1:
                if Lij[i, j] != Lj[j]:
                    return False, (idx[j], idx[Lj[j] - 1])
    return True, (None, None)


def is_conflict_free_farid(D):
    conflict_free = True
    for p in range(D.shape[1]):
        for q in range(p + 1, D.shape[1]):
            oneone = False
            zeroone = False
            onezero = False
            for r in range(D.shape[0]):
                if D[r][p] == 1 and D[r][q] == 1:
                    oneone = True
                if D[r][p] == 0 and D[r][q] == 1:
                    zeroone = True
                if D[r][p] == 1 and D[r][q] == 0:
                    onezero = True
            if oneone and zeroone and onezero:
                conflict_free = False
    return conflict_free


def get_lower_bound_new(noisy, partition_randomly=False):
    def get_important_pair_of_columns_in_conflict(D):
        important_columns = defaultdict(lambda: 0)
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                oneone = 0
                zeroone = 0
                onezero = 0
                for r in range(D.shape[0]):
                    if D[r, p] == 1 and D[r, q] == 1:
                        oneone += 1
                    if D[r, p] == 0 and D[r, q] == 1:
                        zeroone += 1
                    if D[r, p] == 1 and D[r, q] == 0:
                        onezero += 1
                ## greedy approach based on the number of conflicts in a pair of columns
                # if oneone*zeroone*onezero > 0:
                #     important_columns[(p,q)] += oneone*zeroone*onezero
                ## greedy approach based on the min number of 01 or 10 in a pair of columns
                if oneone > 0:
                    important_columns[(p, q)] += min(zeroone, onezero)
        return important_columns

    def get_partition_sophisticated(D):
        ipofic = get_important_pair_of_columns_in_conflict(D)
        if len(ipofic) == 0:
            return []
        sorted_ipofic = sorted(ipofic.items(), key=operator.itemgetter(1), reverse=True)
        pairs = [sorted_ipofic[0][0]]
        elements = [sorted_ipofic[0][0][0], sorted_ipofic[0][0][1]]
        sorted_ipofic.remove(sorted_ipofic[0])
        for x in sorted_ipofic[:]:
            notFound = True
            for y in x[0]:
                if y in elements:
                    sorted_ipofic.remove(x)
                    notFound = False
                    break
            if notFound:
                pairs.append(x[0])
                elements.append(x[0][0])
                elements.append(x[0][1])
        # print(sorted_ipofic, pairs, elements)
        partitions = []
        for x in pairs:
            partitions.append(D[:, x])
        return partitions

    def get_partition_random(D):
        d = int(D.shape[1] / 2)
        partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
        partitions = []
        for x in partitions_id:
            partitions.append(D[:, x])
        return partitions

    def get_lower_bound_for_a_pair_of_columns(D):
        foundOneOne = False
        numberOfZeroOne = 0
        numberOfOneZero = 0
        for r in range(D.shape[0]):
            if D[r, 0] == 1 and D[r, 1] == 1:
                foundOneOne = True
            if D[r, 0] == 0 and D[r, 1] == 1:
                numberOfZeroOne += 1
            if D[r, 0] == 1 and D[r, 1] == 0:
                numberOfOneZero += 1
        if foundOneOne:
            if numberOfZeroOne * numberOfOneZero > 0:
                return min(numberOfZeroOne, numberOfOneZero)
        return 0

    LB = []
    if partition_randomly:
        partitions = get_partition_random(noisy)
    else:
        partitions = get_partition_sophisticated(noisy)
    for D in partitions:
        LB.append(get_lower_bound_for_a_pair_of_columns(D))
    return sum(LB)


def get_lower_bound(noisy, partition_randomly=False):
    def get_important_pair_of_columns_in_conflict(D):
        important_columns = defaultdict(lambda: 0)
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                oneone = 0
                zeroone = 0
                onezero = 0
                for r in range(D.shape[0]):
                    if D[r, p] == 1 and D[r, q] == 1:
                        oneone += 1
                    if D[r, p] == 0 and D[r, q] == 1:
                        zeroone += 1
                    if D[r, p] == 1 and D[r, q] == 0:
                        onezero += 1
                if oneone * zeroone * onezero > 0:
                    important_columns[(p, q)] += oneone * zeroone * onezero
        return important_columns

    def get_partition_sophisticated(D):
        ipofic = get_important_pair_of_columns_in_conflict(D)
        if len(ipofic) == 0:
            return []
        sorted_ipofic = sorted(ipofic.items(), key=operator.itemgetter(1), reverse=True)
        pairs = [sorted_ipofic[0][0]]
        elements = [sorted_ipofic[0][0][0], sorted_ipofic[0][0][1]]
        sorted_ipofic.remove(sorted_ipofic[0])
        for x in sorted_ipofic[:]:
            notFound = True
            for y in x[0]:
                if y in elements:
                    sorted_ipofic.remove(x)
                    notFound = False
                    break
            if notFound:
                pairs.append(x[0])
                elements.append(x[0][0])
                elements.append(x[0][1])
        # print(sorted_ipofic, pairs, elements)
        partitions = []
        for x in pairs:
            partitions.append(D[:, x])
        return partitions

    def get_partition_random(D):
        d = int(D.shape[1] / 2)
        partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
        partitions = []
        for x in partitions_id:
            partitions.append(D[:, x])
        return partitions

    def get_lower_bound_for_a_pair_of_columns(D):
        foundOneOne = False
        numberOfZeroOne = 0
        numberOfOneZero = 0
        for r in range(D.shape[0]):
            if D[r, 0] == 1 and D[r, 1] == 1:
                foundOneOne = True
            if D[r, 0] == 0 and D[r, 1] == 1:
                numberOfZeroOne += 1
            if D[r, 0] == 1 and D[r, 1] == 0:
                numberOfOneZero += 1
        if foundOneOne:
            if numberOfZeroOne * numberOfOneZero > 0:
                return min(numberOfZeroOne, numberOfOneZero)
        return 0

    LB = []
    if partition_randomly:
        partitions = get_partition_random(noisy)
    else:
        partitions = get_partition_sophisticated(noisy)
    for D in partitions:
        LB.append(get_lower_bound_for_a_pair_of_columns(D))
    return sum(LB)


def make_noisy_by_k(data, k):
    data2 = data.copy()
    n, m = np.where(data2 == 1)
    assert k <= len(n), 'k is greater than the number of ones in the input matrix!'
    s = np.random.choice(len(n), k, replace=False)
    assert len(s) == k
    for i in s:
        assert data2[n[i], m[i]] == 1
        data2[n[i], m[i]] = 0
    return data2


def make_noisy_by_fn(data, fn, fp, na):
    def toss(p):
        return True if np.random.random() < p else False
    
    if fn > 1:
        fn = fn / np.count_nonzero(data == 0)
        if fn > 1:
            fn = 0.999
    
    n, m = data.shape
    data2 = -1 * np.ones(shape=(n, m)).astype(int)
    countFP = 0
    countFN = 0
    countNA = 0
    countOneZero = 0
    indexNA = []
    changedBefore = []
    for i in range(n):
        for j in range(m):
            indexNA.append([i, j])
            countOneZero = countOneZero + 1
    random.shuffle(indexNA)
    nas = math.ceil(countOneZero * na)
    for i in range(int(nas)):
        [a, b] = indexNA[i]
        changedBefore.append([a, b])
        data2[a][b] = 2
        countNA = countNA + 1
    for i in range(n):
        for j in range(m):
            if data2[i][j] != 2:
                if data[i][j] == 1:
                    if toss(fn):
                        data2[i][j] = 0
                        countFN = countFN + 1
                    else:
                        data2[i][j] = data[i][j]
                elif data[i][j] == 0:
                    if toss(fp):
                        data2[i][j] = 1
                        countFP = countFP + 1
                    else:
                        data2[i][j] = data[i][j]
    return data2, (countFN, countFP, countNA)


def get_data_by_ms(n, m, seed, fn, fp, na, ms_path=ms_path):
    def build_ground_by_ms(n, m, seed):
        command = "{ms} {n} 1 -s {m} -seeds 7369 217 {r} | tail -n {n}".format(ms=ms_path, n=n, m=m, r=seed)
        result = os.popen(command).read()
        data = np.empty((n, m), dtype=int)
        i = 0
        for s in result.split("\n"):
            j = 0
            for c in list(s):
                data[i, j] = int(c)
                j += 1
            i += 1
        return data

    ground = build_ground_by_ms(n, m, seed)
    if is_conflict_free_farid(ground):
        noisy, (countFN, countFP, countNA) = make_noisy_by_fn(ground, fn, fp, na)
        if not is_conflict_free_farid(noisy) or fn + fp + na == 0:
            return ground, noisy, (countFN, countFP, countNA)
        else:
            return get_data_by_ms(n, m, seed + 1, fn, fp, na, ms_path)
    else:
        print("********************** ERROR ********************")
        print("Ground from ms is not Conflict free!")
        return get_data_by_ms(n, m, seed + 1, fn, fp, na, ms_path)


def is_conflict_free(D):
    conflict_free = True
    for p in range(D.shape[1]):
        for q in range(p + 1, D.shape[1]):
            oneone = False
            zeroone = False
            onezero = False
            for r in range(D.shape[0]):
                if D[r, p] == 1 and D[r, q] == 1:
                    oneone = True
                if D[r, p] == 0 and D[r, q] == 1:
                    zeroone = True
                if D[r, p] == 1 and D[r, q] == 0:
                    onezero = True
            if oneone and zeroone and onezero:
                conflict_free = False
    return conflict_free


def count_flips(I, sol_K=None, sol_Y=None, na_value=2):
    flips_0_1 = 0
    flips_1_0 = 0
    flips_2_0 = 0
    flips_2_1 = 0
    n, m = I.shape
    for i in range(n):
        for j in range(m):
            if sol_K is None or sol_K[j] == 0:
                if I[i][j] == 0 and sol_Y[i][j] == 1:
                    flips_0_1 += 1
                elif I[i][j] == 1 and sol_Y[i][j] == 0:
                    flips_1_0 += 1
                elif I[i][j] == na_value and sol_Y[i][j] == 0:
                    flips_2_0 += 1
                elif I[i][j] == na_value and sol_Y[i][j] == 1:
                    flips_2_1 += 1
    return (flips_0_1, flips_1_0, flips_2_0, flips_2_1)



def top10_bad_entries_in_violations(D):
    def calc_how_many_violations_are_in(D, i, j):
        total = 0
        for p in range(D.shape[1]):
            if p == j:
                continue
            oneone = 0
            zeroone = 0
            onezero = 0
            founded = False
            for r in range(D.shape[0]):
                if D[r, p] == 1 and D[r, j] == 1:
                    oneone += 1
                    if r == i:
                        founded = True
                if D[r, p] == 0 and D[r, j] == 1:
                    zeroone += 1
                    if r == i:
                        founded = True
                if D[r, p] == 1 and D[r, j] == 0:
                    onezero += 1
                    if r == i:
                        founded = True
            if founded:
                total += oneone * zeroone * onezero
        return total

    violations = {}
    for r in range(D.shape[0]):
        for p in range(D.shape[1]):
            if D[r, p] == 0:
                violations[(r, p)] = calc_how_many_violations_are_in(D, r, p)

    for x in sorted(violations.items(), key=operator.itemgetter(1), reverse=True)[:10]:
        print(x[0], "(entry={}): how many gametes".format(D[x[0]]), x[1])


def PhISCS_B_timed(matrix, beta=None, alpha=None, time_limit=3600):
    def returned_PhISCS_B(matrix, returned_dict):
        returned_dict["returned_value"] = PhISCS_B(matrix)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict["returned_value"] = "not filled yet"
    # Start foo as a process
    # multiprocessing.Pool(processes=1)
    p = multiprocessing.Process(target=returned_PhISCS_B, name="returned_PhISCS_B", args=(matrix, return_dict))
    p.start()
    p.join(time_limit)
    # If thread is active
    if p.is_alive():
        p.terminate()
        p.join()
        output_matrix = matrix
        flip_counts = np.zeros(4)
        termination_condition = 'time_limit'
        internal_time = 2 * time_limit
    else:
        output_matrix = return_dict["returned_value"][0]
        flip_counts = return_dict["returned_value"][1]
        termination_condition = 'optimality' #'time_limit'
        internal_time = return_dict["returned_value"][2]
    return output_matrix, flip_counts, termination_condition, internal_time



def rename(new_name):
    def decorator(f):
        f.__name__ = new_name
        return f

    return decorator



def from_interface_to_method(bounding_alg):
    def run_func(x):
        bounding_alg.reset(x)
        ret = bounding_alg.get_bound(sp.lil_matrix(x.shape, dtype=np.int8))
        return ret, bounding_alg.get_name()

    run_func.core = bounding_alg  # include arguments in the name
    run_func.__name__ = bounding_alg.get_name()
    return run_func


def upper_bound_2_sat_timed(matrix, time_limit):
    args = {"matrix" : matrix}
    result = timed_run(upper_bound_2_sat, args, time_limit=time_limit)
    if result["termination_condition"] == "success":
        output = result["output"]
        output = (output[0], output[1], "optimality", output[2])
    elif result["termination_condition"] == "time_limit":
        output = (None, (0,0,0,0), "time_limit", time_limit)
    return output


def zero_or_na(vec, na_value=-1):
    assert is_na_set_correctly(vec, na_value)
    return np.logical_or(vec == 0, vec == na_value)


def get_effective_matrix(I, delta01, delta_na_to_1, change_na_to_0=False):
    x = np.array(I + delta01, dtype=np.int8)
    if delta_na_to_1 is not None:
        na_indices = delta_na_to_1.nonzero()
        x[na_indices] = 1  # should have been (but does not accept): x[na_indices] = delta_na_to_1[na_indices]
    if change_na_to_0:
        x[np.logical_and(x != 0, x != 1)] = 0
    return x


def is_na_set_correctly(matrix, na_value):
    return set(np.unique(matrix)).issubset([0, 1, na_value])



def get_clustering(matrix, na_value=2):
    from sklearn.cluster import MiniBatchKMeans, KMeans
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    X = matrix.T
    X[X == na_value] = 0.5
    # k_means = KMeans(init='k-means++', n_clusters=2, n_init =10)
    k_means = KMeans(init='random', n_clusters=2, n_init=2)
    # k_means = KMeans(init='random', n_clusters=2)
    k_means.fit(X)
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    coloring = np.array(k_means_labels)
    # print(repr(coloring))
    # coloring = np.ones(coloring.shape)
    # coloring = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    # coloring = np.random.randint(2, size=coloring.shape)
    return coloring


def save_matrix_to_file(filename, numpy_input):
    n, m = numpy_input.shape
    dfout = pd.DataFrame(numpy_input)
    dfout.columns = ['mut'+str(j) for j in range(m)]
    dfout.index = ['cell'+str(j) for j in range(n)]
    dfout.index.name = 'cellIDxmutID'
    dfout.to_csv(filename, sep='\t')


def draw_tree_muts_in_edges(filename, add_cells=False):
    bulkfile=''
    addBulk=False

    import numpy as np
    import pandas as pd
    import pygraphviz as pyg
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

    def contains(col1, col2):
        for i in range(len(col1)):
            if not col1[i] >= col2[i]:
                return False
        return True

    df = pd.read_csv(filename, sep='\t', index_col=0)
    splitter_mut = '\n'
    matrix = df.values
    names_mut = list(df.columns)

    i = 0
    while i < matrix.shape[1]:
        j = i + 1
        while j < matrix.shape[1]:
            if np.array_equal(matrix[:,i], matrix[:,j]):
                matrix = np.delete(matrix, j, 1)
                x = names_mut.pop(j)
                names_mut[i] += splitter_mut + x
                j -= 1
            j += 1
        i += 1

    rows = matrix.shape[0]
    cols = matrix.shape[1]
    dimensions = np.sum(matrix, axis=0)
    indices = np.argsort(dimensions)
    dimensions = np.sort(dimensions)
    names_mut = [names_mut[indices[i]] for i in range(cols)]

    G = nx.DiGraph()
    G.add_node(cols)
    G.add_node(cols-1)
    G.add_edge(cols, cols-1, label=names_mut[cols-1])
    node_mud = {}
    node_mud[names_mut[cols-1]] = cols-1

    i = cols - 2
    while i >= 0:
        if dimensions[i] == 0:
            break
        attached = False
        for j in range(i+1, cols):
            if contains(matrix[:, indices[j]], matrix[:, indices[i]]):
                G.add_node(i)
                G.add_edge(node_mud[names_mut[j]], i, label=names_mut[i])
                node_mud[names_mut[i]] = i
                attached = True
                break
        if not attached:
            G.add_node(i)
            G.add_edge(cols, i, label=names_mut[i])
            node_mud[names_mut[i]] = i
        i -=1

    clusters = {}
    for node in G:
        if node == cols:
            G._node[node]['label'] = '<<b>germ<br/>cells</b>>'
            G._node[node]['fontname'] = 'Helvetica'
            G._node[node]['width'] = 0.4
            G._node[node]['style'] = 'filled'
            G._node[node]['penwidth'] = 3
            G._node[node]['fillcolor'] = 'gray60'
            continue
        untilnow_mut = []
        sp = nx.shortest_path(G, cols, node)
        for i in range(len(sp)-1):
            untilnow_mut += G.get_edge_data(sp[i], sp[i+1])['label'].split(splitter_mut)
        untilnow_cell = df.loc[(df[untilnow_mut] == 1).all(axis=1) & \
                               (df[[x for x in df.columns if x not in untilnow_mut]] == 0).all(axis=1)].index
        if len(untilnow_cell) > 0:
            clusters[node] = '\n'.join(untilnow_cell)
        else:
            clusters[node] = '-'
        
        if add_cells:
            G._node[node]['label'] = clusters[node]
        else:
            G._node[node]['label'] = ''
            G._node[node]['shape'] = 'circle'
        G._node[node]['fontname'] = 'Helvetica'
        G._node[node]['width'] = 0.4
        G._node[node]['style'] = 'filled'
        G._node[node]['penwidth'] = 2
        G._node[node]['fillcolor'] = 'gray90'
    i = 1
    for k, v in clusters.items():
        if v == '-':
            clusters[k] = i*'-'
            i += 1

    header = ''
    if addBulk:
        vafs = {}
        bulkMutations = readMutationsFromBulkFile(bulkfile)
        sampleIDs = bulkMutations[0].getSampleIDs()
        for mut in bulkMutations:
            temp_vaf = []
            for sample in sampleIDs:
                temp_vaf.append(str(mut.getVAF(sampleID=sample)))
            vafs[mut.getID()] = '<font color="blue">'+','.join(temp_true)+'</font>'        
        for edge in G.edges():
            temp = []
            for mut in G.get_edge_data(edge[0],edge[1])['label'].split(splitter_mut):
                mut = '<u>' + mut + '</u>' + ': ' + vafs_true[mut] + '; ' + vafs_noisy[mut]
                temp.append(mut)
            temp = '<' + '<br/>'.join(temp) + '>'
            G.get_edge_data(edge[0],edge[1])['label'] = temp

        for mut in bulkMutations:
            try:
                isatype = mut.getINFOEntryStringValue('ISAVtype')
                header += mut.getID() + ': ' + isatype + '<br/>'
            except:
                pass
    
    temp = df.columns[(df==0).all(axis=0)]
    if len(temp) > 0:
        header += 'Became Germline: ' + ','.join(temp) + '<br/>'
    
    '''
    with open(filename[:-len('.CFMatrix')]+'.log') as fin:
        i = 0
        for line in fin:
            i += 1
            if i > 10 and i < 18:
                header += line.rstrip() + '<br/>'
    '''

    H = nx.relabel_nodes(G, clusters)
    html = '''<{}>'''.format(header)
    H.graph['graph'] = {'label':html, 'labelloc':'t', 'resolution':300, 'fontname':'Helvetica', 'fontsize':8}
    H.graph['node'] = {'fontname':'Helvetica', 'fontsize':8}
    H.graph['edge'] = {'fontname':'Helvetica', 'fontsize':8}
    
    mygraph = to_agraph(H)
    mygraph.layout(prog='dot')
    outputpath = filename[:-len('.CFMatrix')]
    mygraph.draw('{}.edges.png'.format(outputpath))


def draw_tree_muts_in_nodes(filename):
    addBulk = False
    bulkfile = ''
    import numpy as np
    import pandas as pd
    import pygraphviz as pyg

    graph = pyg.AGraph(strict=False, directed=True)
    font_name = 'Avenir'

    class Node:
        def __init__(self, name, parent):
            self.name = name
            self.parent = parent
            self.children = []
            if parent:
                parent.children.append(self)

    def print_tree(node):
        graph.add_node(node.name, label=node.name, fontname=font_name, color='black', penwidth=3.5)
        for child in node.children:
            graph.add_edge(node.name, child.name)
            print_tree(child)

    def contains(col1, col2):
        for i in range(len(col1)):
            if not col1[i] >= col2[i]:
                return False
        return True

    def write_tree(matrix, names):
        i = 0
        while i < matrix.shape[1]:
            j = i + 1
            while j < matrix.shape[1]:
                if np.array_equal(matrix[:,i], matrix[:,j]):
                    matrix = np.delete(matrix, j, 1)
                    x = names.pop(j)
                    names[i] += '<br/><br/>' + x
                    j -= 1
                j += 1
            names[i] = '<'+names[i]+'>'
            i += 1

        rows = len(matrix)
        cols = len(matrix[0])
        dimensions = np.sum(matrix, axis=0)
        # ordered indeces
        indeces = np.argsort(dimensions)
        dimensions = np.sort(dimensions)
        mutations_name = []
        for i in range(cols):
            mutations_name.append(names[indeces[i]])

        root = Node(mutations_name[-1], None)
        mut_nod = {}
        mut_nod[mutations_name[cols-1]] = root

        i = cols - 2
        while i >=0:
            if dimensions[i] == 0:
                break
            attached = False
            for j in range(i+1, cols):
                if contains(matrix[:, indeces[j]], matrix[:, indeces[i]]):
                    node = Node(mutations_name[i], mut_nod[mutations_name[j]])
                    mut_nod[mutations_name[i]] = node
                    attached = True
                    break
            if not attached:
                node = Node(mutations_name[i], root)
                mut_nod[mutations_name[i]] = node
            i -=1
        print_tree(root)

    if addBulk:
        vafs = {}
        bulkMutations = readMutationsFromBulkFile(bulkfile)
        sampleIDs = bulkMutations[0].getSampleIDs()
        for mut in bulkMutations:
            temp_vaf = []
            for sample in sampleIDs:
                temp_vaf.append('<font color="blue">' + str(mut.getVAF(sampleID=sample)) + '</font>')
            vafs[mut.getID()] = '{} ({})'.format(mut.getID(), ','.join(temp_vaf))

    inp = np.genfromtxt(filename, skip_header=1, delimiter='\t')
    with open(filename, 'r') as fin:
        if addBulk:
            mutation_names = [vafs[x] for x in fin.readline().strip().split('\t')[1:]]
        else:
            mutation_names = fin.readline().strip().split('\t')[1:]
    sol_matrix = np.delete(inp, 0, 1)
    write_tree(sol_matrix, mutation_names)
    graph.layout(prog='dot')
    outputpath = filename[:-len('.CFMatrix')]
    graph.draw('{}.nodes.png'.format(outputpath))


def infer_na_value(x):
    vals = set(np.unique(x))
    all_vals = copy.copy(vals)
    vals.remove(0)
    vals.remove(1)
    if len(vals) > 0:
        assert len(vals) == 1, "Unable to infer na: There are more than three values:" + repr(all_vals)
        return vals.pop()
    return None


def count_nf(solution, x):
    nf = len(np.where(np.logical_and(solution != x, np.logical_or(x == 0, x == 1)))[0])
    return nf




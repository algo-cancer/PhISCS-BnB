from Utils.const import *
from Utils.interfaces import *
from Utils.instances import *
from Boundings.two_sat import *

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neighbors import KernelDensity


plg = print_line_iter()


def get_one_co_partition(matrix, na_value=2):
    all_rows_the_same = (matrix == matrix[0]).all()
    if all_rows_the_same:
        return np.ones(matrix.shape[0], dtype=np.bool), np.ones(matrix.shape[1], dtype=np.bool)
    row_colors = get_coloring(matrix, k=2, cols=False, na_value=na_value)
    assert len(np.unique(row_colors)) == 2  # rows should be split
    combine = np.hstack((row_colors.reshape(-1, 1), matrix))
    covs = np.cov(combine.T)[0, 1:]
    a = covs.min()
    b = covs.max()
    l = b - a
    unit = l / 10  # justify this 10
    a -= unit
    b += unit
    x_d = np.linspace(a, b, covs.shape[0] * 10)
    done = False
    # mask = covs > ref * 0.9  #todo: better rule
    up = l
    low = l / 1000
    mid = (up + low) / 2

    # mid /= 16
    for i in range(20):  # at most 20 times
        # print("mid=", mid)
        # instantiate and fit the KDE model
        if mid < 1e-5:
            threshold = np.median(covs)
            break
        kde = KernelDensity(bandwidth=mid, kernel='gaussian')
        kde.fit(covs[:, None])
        # score_samples returns the log of the probability density
        logprob = kde.score_samples(x_d[:, None])
        result = np.r_[True, logprob[1:] < logprob[:-1]] & np.r_[logprob[:-1] < logprob[1:], True]
        minima = np.nonzero(result)[0]
        threshold = x_d[minima[-2]]
        cluster_size = np.count_nonzero(covs > threshold)

        if cluster_size < 10:  # too fine
            low = mid
        elif cluster_size < 0.5 * matrix.shape[1]:  # right value
            break
        else:  # too coarse (cluster_size is too big)
            up = mid
        mid = (up + low) / 2
    else:  # if perfect minima was not found
        if len(minima) >= 3:  # a natural choice good enough
            threshold = x_d[minima[-2]]
        else:
            threshold = np.median(covs)

    assert isinstance(threshold, float), str(threshold) + " _ " + str(type(threshold))
    mask = covs > threshold

    cluster_size = np.sum(mask)
    # next(plg)
    while cluster_size == 0 or cluster_size == matrix.shape[1]:
        # next(plg)
        print(mask.size)
        mask = np.random.randint(2, size= mask.size, dtype = np.bool)
        cluster_size = np.sum(mask)
    return row_colors == 1, mask


def get_co_partitioning(matrix, base_n_cols = 30, na_value = 2):
    matrix = matrix.copy()
    ret = []
    # mapping from new matrix indexing to old matrix
    row_map = {a: a for a in range(matrix.shape[0])}
    col_map = {a: a for a in range(matrix.shape[1])}
    left_rows_mask = np.ones(matrix.shape[0], dtype = np.bool)
    left_cols_mask = np.ones(matrix.shape[1], dtype = np.bool)
    while matrix.shape[1] > base_n_cols:
        # next(plg)
        print(matrix.shape)
        rows_mask, cols_mask = get_one_co_partition(matrix, na_value)

        rows_new_mat = np.nonzero(rows_mask)[0]
        rows = [row_map[a] for a in rows_new_mat]
        cols_new_mat = np.nonzero(cols_mask)[0]
        cols = [col_map[a] for a in cols_new_mat]
        left_rows_mask[rows] = False
        left_cols_mask[cols] = False
        ret.append((rows, cols))

        matrix = matrix[np.logical_not(rows_mask), :][:, np.logical_not(cols_mask)]
        row_map_new = {}
        old_ind = 0
        for ind in range(matrix.shape[0]):
            while old_ind in rows_new_mat:
                old_ind += 1
            row_map_new[ind] = row_map[old_ind]
            old_ind += 1

        col_map_new = {}
        old_ind = 0
        for ind in range(matrix.shape[1]):
            while old_ind in cols_new_mat:
                old_ind += 1
            col_map_new[ind] = col_map[old_ind]
            old_ind += 1
        # todo left here
        row_map, col_map = row_map_new, col_map_new

    # next(plg)
    # print(matrix.shape)
    if np.count_nonzero(left_rows_mask) == 0:
        rows = []
    else:
        rows = np.nonzero(left_rows_mask)[0]

    if np.count_nonzero(left_cols_mask) == 0:
        cols = []
    else:
        cols = np.nonzero(left_cols_mask)[0]
    if len(rows) > 0 or len(cols) > 0:
        ret.append((rows, cols))

    # for i in range(len(ret)):
    #     print(i, ret[i])
    return ret


def get_coloring(matrix, k=2, cols=False, na_value = 2):
    X = matrix.copy()
    if cols:
        X = X.T
    X[X == na_value] = 0.5
    k_means = KMeans(init='k-means++', n_clusters=k, n_init =10)
    # k_means = KMeans(init='random', n_clusters=k, n_init=10)
    k_means.fit(X)
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    coloring = np.array(k_means_labels)
    return coloring


def upper_bound_clustering(matrix, threshold, version, base_n_cols = 30, na_value = 2):
    # # # # # # Not an upper_bound yet!
    # next(plg)
    print(matrix.shape)
    if matrix.shape[1] < base_n_cols:
        return upper_bound_2_sat(matrix, threshold, version)


    row_colors = get_coloring(matrix, k=2, cols=False, na_value=na_value)
    combine = np.hstack((row_colors.reshape(-1, 1), matrix))
    covs = np.cov(combine.T)[0, :]

    ref = covs[0]
    covs = covs[1:]
    a = covs.min()
    b = covs.max()
    l = b - a
    unit = l / 10  # justify this 10
    a -= unit
    b += unit
    x_d = np.linspace(a, b, covs.shape[0] * 10)
    done = False
    # mask = covs > ref * 0.9  #todo: better rule
    up = l
    low = l/1000
    mid = (up + low)/2

    # mid /= 16
    for i in range(20):  # at most 20 times
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=mid, kernel='gaussian')
        kde.fit(covs[:, None])
        # score_samples returns the log of the probability density
        logprob = kde.score_samples(x_d[:, None])
        result = np.r_[True, logprob[1:] < logprob[:-1]] & np.r_[logprob[:-1] < logprob[1:], True]
        minima = np.nonzero(result)[0]
        threshold = x_d[minima[-2]]
        cluster_size = np.count_nonzero(covs > threshold)

        if cluster_size < 10:  # too fine
            low = mid
        elif cluster_size < 0.5 * matrix.shape[1]:  # right value
            break
        else:  # too coarse (cluster_size is too big)
            up = mid
        mid = (up + low)/2
    else:  # if perfect minima was not found
        if len(minima) >= 3:  # a natural choice good enough
            threshold = x_d[minima[-2]]
        else:
            threshold = np.median(covs)

    assert isinstance(threshold, float), str(threshold) + " _ " + str(type(threshold))
    mask = covs > threshold

    cluster_size = np.sum(mask)
    if cluster_size == 0 or cluster_size == matrix.shape[1]:
        print("---------------------------- mask zero", threshold, a, b)
        return upper_bound_2_sat(matrix, threshold, version)

    solution_1, cntflip_1, time_1 = upper_bound_2_sat(matrix[:, mask], 0, 0)

    solution_2, cntflip_2, time_2 = upper_bound_clustering(matrix[:, np.logical_not(mask)],
                                                           threshold, version, base_n_cols)

    time_all = time_1 + time_2
    cntflip = tuple(np.add(cntflip_1, cntflip_2))
    solution = matrix.copy()
    solution[:, mask] = solution_1
    solution[:, np.logical_not(mask)] = solution_2

    print(cntflip_1, cntflip_2, cntflip)
    return solution, cntflip, time_all


class Clustering(BoundingAlgAbstract):
    def __init__(self, row_cluster = False, col_cluster= False):
        """
        """
        self.na_support = True
        self.na_value = None  # will be set in the reset()
        self.matrix = None  # will be set in the reset()
        self._times = None  # will be initiated in the reset() and updated get_bound()
        self.row_cluster = row_cluster
        self.col_cluster = col_cluster

    def get_name(self):
        params = [type(self).__name__, self.row_cluster, self.col_cluster]
        params_str = map(str, params)
        return "_".join(params_str)

    def reset(self, matrix):
        self.matrix = matrix  # TODO: make the model here and do small alterations later
        self._times = {"model_preparation_time": 0, "optimization_time": 0}


    def get_init_node(self):
        # todo:
        # make use this clustering for constraints
        # try heirarchical clustering as upper bound ...
        node = pybnb.Node()

        return None
        # solution, cntflip, time = upper_bound_2_sat(
        #     self.matrix, threshold=self.formulation_threshold, version = self.formulation_version )

        nodedelta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == 0))
        node_na_delta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == 2))
        nodeboundVal = self.get_bound(nodedelta, node_na_delta)

        extraInfo = self.get_extra_info()
        print(extraInfo)
        node.state = (nodedelta, extraInfo["icf"], extraInfo["one_pair_of_columns"], nodeboundVal, self.get_state(), node_na_delta)
        node.queue_priority = self.get_priority(
            till_here=-1,
            this_step=-1,
            after_here=-1,
            icf=True)
        return node

    def get_bound(self, delta, delta_na = None):
        self._extraInfo = None
        current_matrix = get_effective_matrix(self.matrix, delta, delta_na)
        has_na = np.any(current_matrix == self.na_value)

        model_time = time.time()
        partitioning = get_co_partitioning(self.matrix)
        row_partition, col_partition = [], []
        for row_par, col_par in partitioning:
            print(len(row_par), len(col_par))
            row_partition.append(row_par)
            col_partition.append(col_par)
        rc2, col_pair, map_f2ij = make_clustered_2sat_model(current_matrix, row_partition, col_partition)
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        opt_time = time.time()
        variables = rc2.compute()  # TODO try different settings from https://pysathq.github.io/docs/html/api/examples/rc2.html
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        result = 0
        for var_ind in range(len(variables)):
            if variables[var_ind] > 0 \
                    and abs(variables[var_ind]) in map_f2ij \
                    and self.matrix[map_f2ij[abs(variables[var_ind])]] == 0:
                result += 1

        # need gustfield
        # self._extraInfo = {
        #     "icf": col_pair == None,
        #     "one_pair_of_columns": col_pair,
        # }

        return result + delta.count_nonzero()

    def get_priority(self, till_here, this_step, after_here, icf=False):
        if icf:
            return self.matrix.shape[0] * self.matrix.shape[1] + 10
        else:
            sgn = np.sign(self.priority_version)
            pv_abs = self.priority_version * sgn
            if pv_abs == 1:
                return sgn * (till_here + this_step + after_here)
            elif pv_abs == 2:
                return sgn * (this_step + after_here)
            elif pv_abs == 3:
                return sgn * (after_here)
            elif pv_abs == 4:
                return sgn * (till_here + after_here)
            elif pv_abs == 5:
                return sgn * (till_here)
            elif pv_abs == 6:
                return sgn * (till_here + this_step)
            elif pv_abs == 7:
                return 0


if __name__ == "__main__":
    # n, m = 25, 10
    # x = np.random.randint(3, size=(n, m))
    # x = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")
    # next(plg)
    for i in range(90, 100, 10):
        x = read_matrix_from_file("../noisy_500/simNo_1-s_4-m_500-n_500-fn_0.1-k_5217.SC.noisy")
        x = x[:i, :i]
        # get_co_partitioning(x, 25, 2)
        # print(get_one_co_partition(x, 2))
        bnd = Clustering(True)
        # next(plg)
        bnd.reset(x)
        # next(plg)
        delta = sp.lil_matrix(x.shape)
        r = bnd.get_bound(delta)

        next(plg)
        solution, cntflip, time_all = PhISCS_B(x, beta=0.97, alpha=0.00001)
        # next(plg)
        print("bnd.get_bound(delta)=", r)
        print("PhISCS_B:", cntflip, time_all)
        print(r == cntflip[0])
    exit(0)

    for i in range(20):
        x[i, i] = 2
    # exit(0)
    # x = x[:3, :]
    # bnd.get_init_node()
    next(plg)
    solution, cntflip, time_al = upper_bound_clustering(x, 0, 0)
    cntflip = list(cntflip)
    cntflip[2] = np.count_nonzero(solution == 2)
    cntflip = tuple(cntflip)
    next(plg)
    print("upper_bound_clustering:", cntflip, time_all)

    bnd = two_sat(True)
    next(plg)
    bnd.reset(x)
    delta = sp.lil_matrix(x.shape)
    result = bnd.get_bound(delta)

    print("bound result:", result)
    next(plg)
    solution, cntflip, time_all = upper_bound_2_sat(x, 0, 0)
    cntflip = list(cntflip)
    cntflip[2] = np.count_nonzero(solution == 2)
    cntflip = tuple(cntflip)
    next(plg)
    print("upper_bound_2_sat:", cntflip, time_all)







    exit(0)
    # x = I_small
    # x = read_matrix_from_file("test2.SC.noisy")
    # x = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")
    # x = np.hstack((x, x, x, x, x, x, x))
    # x = np.vstack((x, x))
    print(x.shape)
    delta = sp.lil_matrix(x.shape)

    algos =[
        two_sat(priority_version=1, formulation_version=0, formulation_threshold=0),
        # two_sat(priority_version=1, formulation_version=1, formulation_threshold=0),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=0),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.2),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.3),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.4),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.5),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.6),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=0.7),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=1),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=1.5),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=2),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=3),
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=4),
    ]
    for algo in algos:
        a = time.time()
        algo.reset(x)
        bnd = algo.get_bound(delta)
        b = time.time()
        print(bnd, b - a, algo.formulation_threshold, algo._times["model_preparation_time"], algo._times["optimization_time"], sep="\t")
        # node = algo.get_init_node()
        # print(node.state[0].count_nonzero())
        # print(bnd, algo._times)
    # print(bnd)
    # print(algo.get_priority(0,0,bnd, False))

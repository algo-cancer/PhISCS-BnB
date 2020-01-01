from utils.const import *
from utils.interfaces import *
from utils.instances import *
from boundings.two_sat import *


class SparseTwoSat(BoundingAlgAbstract):
    def __init__(self, formulation_threshold=0, fn_rate=0.1, probability_threshold=0.01):
        """
        Todo: fix these:
        :param formulation_version: 0 all, 1 cluster
        :param formulation_threshold:
        """
        self.na_support = True
        self.matrix = None
        self.priority_version = -1
        self.formulation_threshold = formulation_threshold
        self.fn_rate = fn_rate
        self.probability_threshold = probability_threshold
        self._times = None

    def get_name(self):
        params = [type(self).__name__,
                  self.formulation_threshold,
                  self.fn_rate,
                  self.probability_threshold
                  ]
        params_str = map(str, params)
        return "_".join(params_str)

    def reset(self, matrix):
        self.matrix = matrix  # make the model here and do small alterations later
        self._times = {"model_preparation_time": 0, "optimization_time": 0}

    # def get_init_node(self):
    #     node = pybnb.Node()
    #     solution, cntflip, time = upper_bound_2_sat(
    #         self.matrix, threshold = self.formulation_threshold, version = self.formulation_version )
    #
    #     nodedelta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == 0))
    #     node_na_delta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == 2))
    #     nodeboundVal = self.get_bound(nodedelta, node_na_delta)
    #
    #     extraInfo = self.get_extra_info()
    #     print(extraInfo)
    #     node.state = (nodedelta, extraInfo["icf"], extraInfo["one_pair_of_columns"], nodeboundVal, self.get_state(), node_na_delta)
    #     node.queue_priority = self.get_priority(
    #         till_here=-1,
    #         this_step=-1,
    #         after_here=-1,
    #         icf=True)
    #     return node

    def get_bound(self, delta, delta_na = None):

        # todo: dynamic?
        # where to get na_value
        na_value = 2
        self._extraInfo = None
        current_matrix = get_effective_matrix(self.matrix, delta, delta_na)
        has_na = np.any(current_matrix == na_value)

        model_time = time.time()
        # rates, intersection_col, decedent_row = compute_relation_matrices(current_matrix,
        #                                                                   na_value=na_value,
        #                                                                   fn_rate=self.fn_rate,
        #                                                                   prob=self.probability_threshold)
        print(time.time()-model_time)
        rc2, col_pair, map_f2ij, map_b2pq = make_2sat_model(
            current_matrix, self.formulation_threshold, coloring=None, eps=0,
            probability_threshold=self.probability_threshold, fn_rate=self.fn_rate)

        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        opt_time = time.time()
        # TODO try different settings from https://pysathq.github.io/docs/html/api/examples/rc2.html
        variables = rc2.compute()
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        result = 0
        for var_ind in range(len(variables)):
            if variables[var_ind] > 0 \
                    and abs(variables[var_ind]) in map_f2ij \
                    and self.matrix[map_f2ij[abs(variables[var_ind])]] == 0:
                result += 1

        if col_pair is not None:  # if we find colision we are sure icf is true, if not, we are not sure.
            self._extraInfo = {
                "icf": False,
                "one_pair_of_columns": col_pair,
            }
        return result + delta.count_nonzero()

    def get_priority(self, till_here, this_step, after_here, icf=False):
        if icf:
            return self.matrix.shape[0] * self.matrix.shape[1] + 10
        else:
            sgn = np.sign(self.priority_version)
            pv_abs = self.priority_version * sgn
            if pv_abs == 1:
                return sgn * (till_here + this_step + after_here)
            else:
                raise NotImplementedError()


if __name__ == "__main__":
    # file_name = "Data/noisy_500/simNo_1-s_4-m_500-n_500.SC.ground"
    # x = read_matrix_from_file(file_name, folder_path=None)
    # exit(0)
    file_name = "Data/noisy_500/simNo_1-s_4-m_500-n_500-fn_0.1-k_5217.SC.noisy"
    # n, m = 500, 500
    # x = np.zeros((n, m), dtype=np.int8)
    # x[:, :m//3] = 1
    # x[:n//2, m//3:(2 * m//3)] = 1
    # x[n//2:, (2 * m//3):] = 1
    #
    # x = make_noisy_by_k(x, 500)
    # print(x)
    # exit(0)
    # x = np.random.randint(3, size=(n, m))
    # x = I_small
    # x = read_matrix_from_file("test2.SC.noisy")
    x = read_matrix_from_file(file_name, folder_path=None)
    # n1 = np.sum(x==1)
    # print(n1)
    # print(5217/n1)
    # exit(0)
    # x = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")
    # x = np.hstack((x, x, x, x, x, x, x))
    # x = np.vstack((x, x))
    print(x.shape)
    delta = sp.lil_matrix(x.shape)

    algos =[
        # two_sat(priority_version=1, formulation_version=0, formulation_threshold=0),
        SparseTwoSat(probability_threshold=0.5),
        SparseTwoSat(probability_threshold=0.3),
        SparseTwoSat(probability_threshold=0.2),
        SparseTwoSat(probability_threshold=0.1),
        SparseTwoSat(probability_threshold=0.05),
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
        print()
        a = time.time()
        algo.reset(x)
        bnd = algo.get_bound(delta)
        b = time.time()
        print(bnd, b - a, algo.formulation_threshold, algo._times["model_preparation_time"], algo._times["optimization_time"], sep="\t")
        print(algo.get_name(), "bnd=", bnd)
        # node = algo.get_init_node()
        # print(node.state[0].count_nonzero())
        # print(bnd, algo._times)
    # print(bnd)
    # print(algo.get_priority(0,0,bnd, False))

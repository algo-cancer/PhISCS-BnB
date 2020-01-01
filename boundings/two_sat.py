from utils.const import *
from utils.interfaces import *
from utils.instances import *
from algorithms.twosat import *


class TwoSatBounding(BoundingAlgAbstract):
    def __init__(self, priority_version=-1,
                 cluster_rows=False, cluster_cols=False, only_descendant_rows=False,
                 na_value=None,
                 heuristic_setting=None,
                 n_levels=2, eps=0, compact_formulation=False):
        """
        :param priority_version:
        """
        assert not cluster_rows, "Not implemented yet"
        assert not cluster_cols, "Not implemented yet"
        assert not only_descendant_rows, "Not implemented yet"

        self.priority_version = priority_version

        self.na_support = True
        self.na_value = na_value
        self.matrix = None
        self._times = None
        self.next_lb = None
        self.heuristic_setting = heuristic_setting
        self.n_levels = n_levels
        self.eps = eps  # only for upperbound
        self.compact_formulation = compact_formulation
        self.cluster_rows = cluster_rows
        self.cluster_cols = cluster_cols
        self.only_descendant_rows = only_descendant_rows

    def get_name(self):
        params = [type(self).__name__,
                  self.priority_version,
                  self.heuristic_setting,
                  self.n_levels,
                  self.eps,
                  self.compact_formulation
                  ]
        params_str = map(str, params)
        return "_".join(params_str)

    def reset(self, matrix):
        self.matrix = matrix  # todo: make the model here and do small alterations later
        assert is_na_set_correctly(self.matrix, self.na_value)
        # self.na_value = infer_na_value(matrix)
        self._times = {"model_preparation_time": 0, "optimization_time": 0}

    def get_init_node(self):

        # def twosat_solver(matrix, cluster_rows=False, cluster_cols=False, only_descendant_rows=False,
        #                   na_value=None, leave_nas_if_zero=False, return_lb=False, heuristic_setting=None,
        #                   n_levels=2, eps=0, compact_formulation=True):
        #     pass

        node = pybnb.Node()
        solution, model_time, opt_time, lb = twosat_solver(
            self.matrix,
            cluster_rows=self.cluster_rows,
            cluster_cols=self.cluster_cols,
            only_descendant_rows=self.only_descendant_rows,
            na_value=self.na_value,
            leave_nas_if_zero=True,
            return_lb=True,
            heuristic_setting=None,
            n_levels=self.n_levels,
            eps=self.eps,
            compact_formulation=self.compact_formulation
        )
        self._times["model_preparation_time"] += model_time
        self._times["optimization_time"] += opt_time

        nodedelta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == 0))
        node_na_delta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == self.na_value))
        node.state = (nodedelta, True, None, nodedelta.count_nonzero(), self.get_state(), node_na_delta)
        node.queue_priority = self.get_priority(
            till_here=-1,
            this_step=-1,
            after_here=-1,
            icf=True)
        self.next_lb = lb
        return node

    def get_bound(self, delta, delta_na = None):
        # make this dynamic when more nodes were getting explored
        if self.next_lb is not None:
            lb = self.next_lb
            self.next_lb = None
            return lb
        self._extraInfo = None
        current_matrix = get_effective_matrix(self.matrix, delta, delta_na)
        has_na = np.any(current_matrix == self.na_value)

        model_time = time.time()
        return_value = make_constraints_np_matrix(current_matrix, n_levels=self.n_levels, na_value=self.na_value,
                                                  compact_formulation=self.compact_formulation)
        F, map_f2ij, zero_vars, na_vars, hard_constraints, col_pair = \
            return_value.F, return_value.map_f2ij, return_value.zero_vars, return_value.na_vars, \
            return_value.hard_constraints, return_value.col_pair

        if col_pair is not None:
            icf = False
        elif return_value.complete_version:
            icf = True
        else:
            icf = None  # not sure
        rc2 = make_twosat_model_from_np(hard_constraints, F, zero_vars, na_vars,
                                        eps=0,
                                        heuristic_setting=self.heuristic_setting,
                                        compact_formulation=self.compact_formulation)

        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        opt_time = time.time()
        variables = rc2.compute()
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        result = 0
        for var_ind in range(len(variables)):
            if variables[var_ind] > 0 \
                    and abs(variables[var_ind]) in map_f2ij \
                    and self.matrix[map_f2ij[abs(variables[var_ind])]] == 0:
                result += 1

        assert has_na or ((result == 0) == (col_pair is None)), f"{result}_{col_pair}"
        self._extraInfo = {
            "icf": icf,
            "one_pair_of_columns": col_pair,
        }
        ret = result + delta.count_nonzero()
        # print("lb=", ret, result)
        return ret


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
        assert False, "get_priority did not return anything!"


if __name__ == "__main__":
    n, m = 6, 6
    x = np.random.randint(2, size=(n, m))
    # x = I_small
    # x = read_matrix_from_file("test2.SC.noisy")
    # x = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")
    # x = np.hstack((x, x, x, x, x, x, x))
    # x = np.vstack((x, x))
    print(x.shape)
    delta = sp.lil_matrix(x.shape)

    algos =[
        TwoSatBounding()
        # TwoSatBounding(heuristic_setting=None, n_levels=1, compact_formulation=False),
        # TwoSatBounding(heuristic_setting=None, n_levels=2, compact_formulation=False),
        # TwoSatBounding(heuristic_setting=None, n_levels=1, compact_formulation=True),
        # TwoSatBounding(heuristic_setting=None, n_levels=2, compact_formulation=True),
        # TwoSatBounding(heuristic_setting=[True, True, False, True, True], n_levels=2, compact_formulation=False),
    ]
    for algo in algos:
        a = time.time()
        algo.reset(x)
        # bnd = algo.get_bound(delta)
        b = time.time()
        # print(bnd, b - a, algo.formulation_threshold, algo._times["model_preparation_time"], algo._times["optimization_time"], sep="\t")
        print(algo.only_descendant_rows)
        node = algo.get_init_node()
        # print(node.state)
        # print(node.queue_priority)
        # print(algo.get_name(), bnd, algo._times, b - a)
    # print(bnd)
    # print(algo.get_priority(0,0,bnd, False))

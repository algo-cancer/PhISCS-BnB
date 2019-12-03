from Utils.const import *
from Utils.interfaces import *
from operator import add
from functools import reduce
from Utils.instances import *

class two_sat(BoundingAlgAbstract):
    def __init__(self, priority_version = 0, formulation_version = 0, formulation_threshold = 0):
        """
        :param priority_version:
        :param formulation_version: 0 all, 1 cluster
        :param formulation_threshold:
        """
        self.matrix = None
        self.priority_version = priority_version
        self.formulation_version = formulation_version
        self.formulation_threshold = formulation_threshold
        self._times = None

    def get_name(self):
        params = [type(self).__name__, self.priority_version, self.formulation_version,self.formulation_threshold]
        params_str = map(str, params)
        return "_".join(params_str)

    def reset(self, matrix):
        self.matrix = matrix # make the model here and do small alterations later
        self._times = {"model_preparation_time": 0, "optimization_time": 0}

    def get_init_node(self):
        node = pybnb.Node()
        solution, cntflip, time = upper_bound_2_sat(
            self.matrix, threshold = self.formulation_threshold, version = self.formulation_version )

        nodedelta = sp.lil_matrix(solution != self.matrix)
        nodeboundVal = self.get_bound(nodedelta)
        extraInfo = self.get_extra_info()
        node.state = (nodedelta, extraInfo["icf"], extraInfo["one_pair_of_columns"], nodeboundVal, self.get_state())
        node.queue_priority = self.get_priority(
            till_here=-1,
            this_step=-1,
            after_here=-1,
            icf=True)
        return node

    def get_bound(self, delta):
        # print(delta)
        self._extraInfo = None
        current_matrix = np.array(self.matrix + delta) # todo: make this dynamic

        model_time = time.time()
        if self.formulation_version == 0:
            coloring = None
        elif self.formulation_version == 1:
            coloring = get_clustering(current_matrix)
        else:
            raise NotImplementedError("version?")
        rc2, col_pair, map_f2ij, map_b2pq = make_2sat_model(
            current_matrix, self.formulation_threshold, coloring)
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time


        opt_time = time.time()
        variables = rc2.compute() # TODO try different settings from https://pysathq.github.io/docs/html/api/examples/rc2.html
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        result = 0
        for var_ind in range(len(variables)):
            if variables[var_ind] > 0 and abs(variables[var_ind]) in map_f2ij:
                result += 1


        assert self.formulation_version!=0 or ((result == 0) == (col_pair is None)), f"{result}_{col_pair}"
        self._extraInfo = {
            "icf": col_pair == None,
            "one_pair_of_columns": col_pair,
        }
        return result


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
    # n, m = 25, 25
    # x = np.random.randint(2, size=(n, m))
    # x = I_small
    # x = read_matrix_from_file("test2.SC.noisy")
    x = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")
    # x = np.hstack((x, x, x, x, x, x, x))
    # x = np.vstack((x, x))
    print(x.shape)
    delta = sp.lil_matrix(x.shape)

    algos =[
        two_sat(priority_version=1, formulation_version=0, formulation_threshold=0),
        two_sat(priority_version=1, formulation_version=1, formulation_threshold=0),
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
        # bnd = algo.get_bound(delta)
        # b = time.time()
        # print(bnd, b - a, algo.formulation_threshold, algo._times["model_preparation_time"], algo._times["optimization_time"], sep="\t")
        node = algo.get_init_node()
        print(node.state[0].count_nonzero())
        # print(bnd, algo._times)
    # print(bnd)
    # print(algo.get_priority(0,0,bnd, False))

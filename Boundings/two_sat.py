from Utils.const import *
from Utils.interfaces import *
from operator import add
from functools import reduce
from Utils.instances import *

class two_sat(BoundingAlgAbstract):
    def __init__(self, priority_version = 0, formulation_version = 0):
        self.matrix = None
        self.priority_version = priority_version
        self.formulation_version = formulation_version


    def get_name(self):
        return f"{type(self).__name__}_{self.priority_version}_{self.formulation_version}"

    def reset(self, matrix):
        self.matrix = matrix # make the model here and do small alterations later
        self._times = {"model_preparation_time": 0, "optimization_time": 0}

    def get_init_node(self):
        node = pybnb.Node()
        solution, cntflip, time = PhISCS_B_2_sat(self.matrix, version = self.formulation_version)

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
        rc2, col_pair, _, num_var_F = make_2sat_model(current_matrix, self.formulation_version)
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time


        opt_time = time.time()
        variables = rc2.compute()
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        result = 0
        for var_ind in range(num_var_F):
            if variables[var_ind] > 0:
                result += 1


        assert (result == 0) == (col_pair is None), f"{result}_{col_pair}"
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
    # n, m = 15, 15
    # x = np.random.randint(2, size=(n, m))
    x = I_small
    # x = read_matrix_from_file("test2.SC.noisy")
    x = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")
    delta = sp.lil_matrix(x.shape)

    algo = two_sat(priority_version=1, formulation_version=0)
    algo.reset(x)
    bnd = algo.get_bound(delta)

    print(algo._times)
    print(bnd)
    print(algo.get_priority(0,0,bnd, False))

    print()
    algo = two_sat(priority_version=1, formulation_version=1)
    algo.reset(x)
    bnd = algo.get_bound(delta)

    print(algo._times)
    print(bnd)
    print(algo.get_priority(0, 0, bnd, False))
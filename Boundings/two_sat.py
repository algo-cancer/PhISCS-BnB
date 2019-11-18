from Utils.const import *
from Utils.interfaces import *
from operator import add
from functools import reduce
from Utils.instances import *

class two_sat(BoundingAlgAbstract):
    def __init__(self, priority_version = 0):
        self.matrix = None
        self.priority_version = priority_version


    def get_name(self):
        return f"{type(self).__name__}_{self.priority_version}"

    def reset(self, matrix):
        self.matrix = matrix # make the model here and do small alterations later
        self._times = {"model_preparation_time": 0, "optimization_time": 0}

    def get_init_node(self):
        node = pybnb.Node()
        solution, cntflip, time = PhISCS_B_2_sat(self.matrix)

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
        rc2, col_pair = two_sat.make_model(current_matrix)
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time


        opt_time = time.time()
        variables = rc2.compute()
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        result = 0
        for var_ind in range(len(variables)):
            if variables[var_ind] > 0:
                result += 1

        # p = col_pair[0]
        # q = col_pair[1]
        # matrix = current_matrix
        # r01 = np.nonzero(np.logical_and(matrix[:, p] == 0, matrix[:, q] == 1))[0]
        # r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 0))[0]
        # cost = min(len(r01), len(r10))
        #
        # print(r01, r10, cost)
        assert (result == 0) == (col_pair is None), f"{result}_{col_pair}"
        self._extraInfo = {
            "icf": col_pair == None,
            "one_pair_of_columns": col_pair,
        }
        return result

    @staticmethod
    def make_model(matrix):
        rc2 = RC2(WCNF())
        n, m = matrix.shape

        F = np.empty((n, m), dtype=np.int64)
        num_var_F = 0
        map_f2ij = {}
        for i in range(n):
            for j in range(m):
                if matrix[i, j] == 0:
                    num_var_F += 1
                    map_f2ij[num_var_F] = (i, j)
                    F[i, j] = num_var_F
                    rc2.add_clause([-F[i,j]], weight = 1)

        col_pair = None
        pair_cost = 0
        for p in range(m):
            for q in range(m):
                if p != q and np.any(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 1)):
                    r01 = np.nonzero(np.logical_and(matrix[:, p] == 0, matrix[:, q] == 1))[0]
                    r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 0))[0]
                    cost = min(len(r01), len(r10))
                    if cost > pair_cost:
                        col_pair = (p, q)
                        pair_cost = cost
                    for a, b in itertools.product(r01, r10):
                        rc2.add_clause([F[a, p], F[b, q]]) # at least one of them should be flipped
        return rc2, col_pair

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
    # x = I_small
    x = read_matrix_from_file()
    delta = sp.lil_matrix(x.shape)

    algo = two_sat()
    algo.reset(x)
    bnd = algo.get_bound(delta)

    print(algo._times)
    print(bnd)
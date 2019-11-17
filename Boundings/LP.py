from Utils.const import *
from Utils.interfaces import *
from Utils.instances import *
from Utils.util import *


# class DynamicLPBounding(BoundingAlgAbstract):
#   def __init__(self, ratio=None):
#     raise NotImplementedError("The method not implemented")


class SemiDynamicLPBounding(BoundingAlgAbstract):
    def __init__(
        self,
        ratio=None,
        continuous=True,
        n_threads: int = 1,
        tool="Gurobi",
        priority_sign=-1,
        change_bound_method=True,
        for_loop_constrs=True,
        add_extra_constraints=False,
    ):
        """
        :param ratio:
        :param continuous:
        :param n_threads:
        :param tool: in ["Gurobi", "ORTools"]
        """
        super().__init__()
        self.ratio = ratio
        self.model = None
        self.y_vars = None
        self.continuous = continuous
        self.n_threads = n_threads
        self.tool = tool
        self.priority_sign = priority_sign
        self.change_bound_method = change_bound_method
        self.for_loop_constrs = for_loop_constrs
        self.add_extra_constraints = add_extra_constraints


    def get_name(self):
        return (
            f"{type(self).__name__}_{self.ratio}_{self.continuous}"
            f"_{self.tool}_{self.priority_sign}_{self.change_bound_method}_{self.for_loop_constrs}"
        )

    def reset(self, matrix):
        self._times = {"model_preparation_time": 0, "optimization_time": 0}
        self.matrix = matrix

        model_time = time.time()
        if self.tool == "Gurobi":
            self.model, self.y_vars = StaticLPBounding.make_Gurobi_model(self.matrix, continuous=self.continuous
                                                                         , add_extra_constraints=self.add_extra_constraints)
        elif self.tool == "ORTools":
            self.model, self.y_vars = StaticLPBounding.make_OR_tools_model(self.matrix, continuous=self.continuous)
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        optTime = time.time()
        if self.tool == "Gurobi":
            self.model.optimize()
        elif self.tool == "ORTools":
            self.model.Solve()
        optTime = time.time() - optTime
        self._times["optimization_time"] += optTime

    def get_bound(self, delta):
        self._extraInfo = None
        flips = np.transpose(delta.nonzero())

        model_time = time.time()
        new_constrs = (self.y_vars[flips[i, 0], flips[i, 1]] == 1 for i in range(flips.shape[0]))
        new_constrs_returned = []
        if self.tool == "Gurobi":
            if self.change_bound_method:
                for i in range(flips.shape[0]):
                    self.y_vars[flips[i, 0], flips[i, 1]].lb = 1
            elif self.for_loop_constrs:
                for cns in new_constrs:
                    new_constrs_returned.append(self.model.addConstr(cns))
            else:
                new_constrs_returned = self.model.addConstrs(new_constrs)
        elif self.tool == "OR_tools":
            for constraint in new_constrs:
                self.model.Add(constraint)
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        obj_val = None
        opt_time = time.time()
        if self.tool == "Gurobi":
            self.model.optimize()
            obj_val = np.int(np.ceil(self.model.objVal))
        elif self.tool == "OR_tools":
            self.model.Solve()
            obj_val = self.model.Objective().Value()
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        if self.ratio is not None:
            bound = np.int(np.ceil(self.ratio * obj_val))
        else:
            bound = np.int(np.ceil(obj_val))

        model_time = time.time()
        if self.tool == "Gurobi":
            if self.change_bound_method:
                for i in range(flips.shape[0]):
                    self.y_vars[flips[i, 0], flips[i, 1]].lb = 0
            else:
                if self.for_loop_constrs:
                    for constraint in new_constrs_returned:
                        self.model.remove(constraint)
                else:
                    for constraint in new_constrs_returned.values():
                        self.model.remove(constraint)
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        return bound

    def has_state(self):
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if not isinstance(self.y_vars[i, j], np.int):
                    return hasattr(self.y_vars[i, j], "X")

    def get_priority(self, new_bound, icf=False):
        if icf:
            return 1000
        else:
            return new_bound * self.priority_sign


class StaticLPBounding(BoundingAlgAbstract):
    def __init__(self, ratio=None, continuous=True):
        super().__init__()
        self.ratio = ratio
        self.matrix = None
        self.continuous = continuous

    def get_name(self):
        return type(self).__name__ + f"_{self.ratio}_{self.continuous}"

    def reset(self, matrix):
        self.matrix = matrix

    def getBound(self, delta):
        bound = StaticLPBounding.LP_brief(self.matrix + delta, self.continuous)
        if self.ratio is not None:
            bound = np.int(np.ceil(self.ratio * bound))
        else:
            bound = np.int(np.ceil(bound))

        return bound + delta.count_nonzero()

    @staticmethod
    def LP_brief(matrix, continuous=True):
        model, Y = StaticLPBounding.make_Gurobi_model(matrix, continuous=continuous)
        return StaticLPBounding.LP_Bounding_From_Model(model)

    @staticmethod
    def LP_Bounding_From_Model(model):
        model.optimize()
        return np.int(np.ceil(model.objVal))

    @staticmethod
    def make_OR_tools_model(matrix, continuous=True):
        model = pywraplp.Solver(f"LP_ORTools_{time.time()}", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        num_cells = matrix.shape[0]
        num_mutations = matrix.shape[1]
        Y = {}
        for c in range(num_cells):
            for m in range(num_mutations):
                if matrix[c, m] == 0:
                    Y[c, m] = model.NumVar(0, 1, "Y({0},{1})".format(c, m))
                elif matrix[c, m] == 1:
                    Y[c, m] = 1
        B = {}
        for p in range(num_mutations):
            for q in range(num_mutations):
                B[p, q, 1, 1] = model.NumVar(0, 1, "B[{0},{1},1,1]".format(p, q))
                B[p, q, 1, 0] = model.NumVar(0, 1, "B[{0},{1},1,0]".format(p, q))
                B[p, q, 0, 1] = model.NumVar(0, 1, "B[{0},{1},0,1]".format(p, q))
        for i in range(num_cells):
            for p in range(num_mutations):
                for q in range(num_mutations):
                    model.Add(Y[i, p] + Y[i, q] - B[p, q, 1, 1] <= 1)
                    model.Add(-Y[i, p] + Y[i, q] - B[p, q, 0, 1] <= 0)
                    model.Add(Y[i, p] - Y[i, q] - B[p, q, 1, 0] <= 0)
        for p in range(num_mutations):
            for q in range(num_mutations):
                model.Add(B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2)

        objective = sum([Y[c, m] for c in range(num_cells) for m in range(num_mutations)])
        model.Minimize(objective)
        return model, Y

    @staticmethod
    def make_Gurobi_model(I, continuous=True, add_extra_constraints=False):
        if continuous:
            varType = GRB.CONTINUOUS
        else:
            varType = GRB.BINARY

        num_cells, num_mutations = I.shape

        model = Model(f"LP_Gurobi_{time.time()}")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        Y = {}
        for c in range(num_cells):
            for m in range(num_mutations):
                if I[c, m] == 0:
                    Y[c, m] = model.addVar(0, 1, obj=1, vtype=varType, name="Y({0},{1})".format(c, m))
                elif I[c, m] == 1:
                    Y[c, m] = 1
        # TODO: Working here!
        B = {}
        for p in range(num_mutations):
            for q in range(num_mutations):
                B[p, q, 1, 1] = model.addVar(0, 1, vtype=varType, obj=0, name="B[{0},{1},1,1]".format(p, q))
                B[p, q, 1, 0] = model.addVar(0, 1, vtype=varType, obj=0, name="B[{0},{1},1,0]".format(p, q))
                B[p, q, 0, 1] = model.addVar(0, 1, vtype=varType, obj=0, name="B[{0},{1},0,1]".format(p, q))

        for i in range(num_cells):
            for p in range(num_mutations):
                for q in range(num_mutations):
                    model.addConstr(Y[i, p] + Y[i, q] - B[p, q, 1, 1] <= 1)
                    model.addConstr(-Y[i, p] + Y[i, q] - B[p, q, 0, 1] <= 0)
                    model.addConstr(Y[i, p] - Y[i, q] - B[p, q, 1, 0] <= 0)

        for p in range(num_mutations):
            for q in range(num_mutations):
                model.addConstr(B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2)

        # new constraints
        if add_extra_constraints:
            for p in range(num_mutations):
                for q in range(num_mutations):
                    if p != q and np.any(np.logical_and(I[:, p] == 1, I[:, q] == 1)):
                        r01 = np.nonzero(np.logical_and(I[:, p] == 0, I[:, q] == 1))[0]
                        r10 = np.nonzero(np.logical_and(I[:, p] == 1, I[:, q] == 0))[0]
                        for a, b in itertools.product(r01, r10):
                            model.addConstr(Y[a,p] + Y[b,q] >= 1)



        model.Params.ModelSense = GRB.MINIMIZE
        return model, Y


class StaticILPBounding(BoundingAlgAbstract):
    def __init__(self, ratio=None):
        super().__init__()
        self.ratio = ratio
        self.matrix = None

    def reset(self, matrix):
        self.matrix = matrix

    def getBound(self, delta):
        model, Y = StaticLPBounding.make_Gurobi_model(self.matrix + delta, continuous=False)
        optim = StaticLPBounding.LP_Bounding_From_Model(model)
        if self.ratio is not None:
            bound = np.int(np.ceil(self.ratio * optim))
        else:
            bound = np.int(np.ceil(optim))
        return bound + delta.count_nonzero()


if __name__ == "__main__":

    # n, m = 15, 15
    # x = np.random.randint(2, size=(n, m))
    # x = I6
    x = read_matrix_from_file()
    delta = sp.lil_matrix(x.shape)
    # print(x)
    # StaticLPBounding.make_Gurobi_model(x)


    algoF = SemiDynamicLPBounding(ratio=None, continuous=True, n_threads=1, tool="Gurobi", priority_sign=-1, add_extra_constraints=False)
    algoT = SemiDynamicLPBounding(ratio=None, continuous=True, n_threads=1, tool="Gurobi", priority_sign=-1, add_extra_constraints=True)

    reset_time = time.time()
    algoF.reset(x)
    reset_time = time.time() - reset_time
    print(reset_time)

    reset_time = time.time()
    algoT.reset(x)
    reset_time = time.time() - reset_time
    print(reset_time)

    print(len(algoF.model.getConstrs()))
    print(len(algoT.model.getConstrs()))

    xp = np.asarray(x + delta)
    optim = myPhISCS_I(xp)

    print("Optimal answer:", optim)

    bound_time = time.time()
    bft = algoF.get_bound(delta)
    bound_time = time.time() - bound_time
    print(bound_time)

    bound_time = time.time()
    btt = algoT.get_bound(delta)
    bound_time = time.time() - bound_time
    print(bound_time)

    print(bft, btt)
    exit(0)
    # algo.model.reset()
    algoPrim = algo.model.copy()

    print(algo.has_state(), algoPrim.has_state())
    print(StaticLPBounding.LP_brief(xp), algo.get_bound(delta))

    for t in range(5):
        ind = np.nonzero(1 - (x + delta))
        a, b = ind[0][0], ind[1][0]
        delta[a, b] = 1
        print(algo.has_state())
        # algo.model.reset()
        calc_time = time.time()
        bnd_adapt = algo.get_bound(delta)
        calc_time = time.time() - calc_time
        # print(calcTime)
        algo.get_bound(delta)
        bnd_full = StaticLPBounding.LP_brief(x + delta) + t + 1
        # print(bndFull == bndAdapt, bndFull, bndAdapt)
        static_LP_bounding_bnd = static_LP_bounding.getBound(delta)
        # print(bndFull == staticLPBoundingBnd, staticLPBoundingBnd)


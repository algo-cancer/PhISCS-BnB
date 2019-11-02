from Utils.const import *
from Utils.interfaces import *
from Utils.util import *
from Utils.instances import *
import collections


Constraint = collections.namedtuple('Constraint', " ".join(('type', 'row', 'col1', 'col2', 'exp')))

class SubsampleLPBounding(BoundingAlgAbstract):
    def __init__(
        self,
        ratio=None,
        continuous=True,
        n_threads: int = 1,
        priority_sign=-1,
        strength = 1,
    ):
        """
        :param ratio:
        :param continuous:
        :param n_threads:
        """
        super().__init__()
        self.ratio = ratio
        self.model = None
        self.y_vars = None
        self.b_vars = None
        self.continuous = continuous
        self.n_threads = n_threads
        self.priority_sign = priority_sign
        self.constraints = None
        self.weights = None
        self.strength = strength

    def get_name(self):
        return (
            f"{type(self).__name__}_{self.strength}_{self.continuous}"
            f"_{self.priority_sign}"
        )

    def reset(self, matrix):
        self._times = {"model_preparation_time": 0, "optimization_time": 0}
        self.matrix = matrix
        model_time = time.time()
        self.model, self.y_vars, self.b_vars = StaticUtil.make_Gurobi_model(
            matrix, continuous=True, add_constraints=False)
        self.constraints = StaticUtil.get_constraints(matrix.shape[0], matrix.shape[1], self.y_vars, self.b_vars)
        if self.strength is not None:
            siz = int(len(self.constraints)*self.strength)
            sample = np.random.choice(len(self.constraints), siz, replace = False)
        else:
            sample = range(len(self.constraints))
        self.add_constraints(self.constraints, sample)
        self.weights = np.ones(len(self.constraints))
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        optTime = time.time()
        self.model.optimize()
        optTime = time.time() - optTime
        self._times["optimization_time"] += optTime

    def get_bound(self, delta):
        self._extraInfo = None
        flips = np.transpose(delta.nonzero())
        model_time = time.time()
        for i in range(flips.shape[0]):
            self.y_vars[flips[i, 0], flips[i, 1]].lb = 1
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time
        obj_val = None
        opt_time = time.time()
        self.model.optimize()
        obj_val = np.int(np.ceil(self.model.objVal))
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        if self.ratio is not None:
            bound = np.int(np.ceil(self.ratio * obj_val))
        else:
            bound = np.int(np.ceil(obj_val))

        model_time = time.time()
        for i in range(flips.shape[0]):
            self.y_vars[flips[i, 0], flips[i, 1]].lb = 0
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        return bound

    def get_priority(self, new_bound, icf=False):
        if icf:
            return 1000
        else:
            return new_bound * self.priority_sign


    def add_constraints(self, constraints, indices):
        for ind in indices:
            self.model.addConstr(constraints[ind].exp)

class StaticUtil():
    def __init__(self):
        pass

    @staticmethod
    def get_constraints(num_cells, num_mutations, Y, B):
        ret = []

        for i in range(num_cells):
            for p in range(num_mutations):
                for q in range(num_mutations):
                    ret.append(Constraint(type="B11", row = i, col1=p, col2=q,
                                          exp=Y[i, p] + Y[i, q] - B[p, q, 1, 1] <= 1))
                    ret.append(Constraint(type="B01", row = i, col1=p, col2=q,
                                          exp=-Y[i, p] + Y[i, q] - B[p, q, 0, 1] <= 0))
                    ret.append(Constraint(type="B10", row = i, col1=p, col2=q,
                                          exp=Y[i, p] - Y[i, q] - B[p, q, 1, 0] <= 0))

        for p in range(num_mutations):
            for q in range(num_mutations):
                ret.append(Constraint(type="general", row = None, col1=p, col2=q,
                                      exp=B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2))
        return ret


    @staticmethod
    def make_Gurobi_model(I, continuous=True, add_constraints = True):
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

        B = {}
        for p in range(num_mutations):
            for q in range(num_mutations):
                B[p, q, 1, 1] = model.addVar(0, 1, vtype=varType, obj=0, name="B[{0},{1},1,1]".format(p, q))
                B[p, q, 1, 0] = model.addVar(0, 1, vtype=varType, obj=0, name="B[{0},{1},1,0]".format(p, q))
                B[p, q, 0, 1] = model.addVar(0, 1, vtype=varType, obj=0, name="B[{0},{1},0,1]".format(p, q))

        if add_constraints:
            for i in range(num_cells):
                for p in range(num_mutations):
                    for q in range(num_mutations):
                        model.addConstr(Y[i, p] + Y[i, q] - B[p, q, 1, 1] <= 1)
                        model.addConstr(-Y[i, p] + Y[i, q] - B[p, q, 0, 1] <= 0)
                        model.addConstr(Y[i, p] - Y[i, q] - B[p, q, 1, 0] <= 0)

            for p in range(num_mutations):
                for q in range(num_mutations):
                    model.addConstr(B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2)

        model.Params.ModelSense = GRB.MINIMIZE
        return model, Y, B



if __name__ == "__main__":
    n, m = 50, 50
    x = np.random.randint(2, size=(n, m))
    # x = I6
    # x = I_small
    delta = sp.lil_matrix(x.shape)

    # xx = StaticUtil.get_constraints(n, m)
    # print(len(xx))
    # exit(0)
    for strength in list([(r + 1) / 10 for r in range(7)]) + [None,]:
        subsample_bounding = SubsampleLPBounding(strength= strength)
        subsample_bounding.reset(x)
        print(strength, subsample_bounding.get_bound(delta), subsample_bounding.times)
    exit(0)
    static_LP_bounding = StaticLPBounding()
    static_LP_bounding.reset(x)

    algo = SemiDynamicLPBounding(ratio=None, continuous=True, n_threads=1, tool="Gurobi", priority_sign=-1)
    reset_time = time.time()
    algo.reset(x)
    reset_time = time.time() - reset_time
    print(reset_time)

    xp = np.asarray(x + delta)
    optim = myPhISCS_I(xp)

    print("Optimal answer:", optim)
    print(StaticLPBounding.LP_brief(xp), algo.get_bound(delta))
    print(algo.has_state())
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


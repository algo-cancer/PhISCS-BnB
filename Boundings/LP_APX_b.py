from Utils.const import *
from Utils.interfaces import *
from Utils.util import *
from Utils.instances import *
import collections


Constraint = collections.namedtuple('Constraint', " ".join(('type', 'row', 'col1', 'col2', 'exp')))

class SubsampleLPBounding_b(BoundingAlgAbstract):
    @staticmethod
    def n_const(n, m):
        return (0.35301756  * n + 5.71581769) * (1.05 * m * m + 35.54995381 * m - 150.87030838)

    def __init__(self, n_cnst_func=None):
        """
        :param ratio:
        :param continuous:
        :param n_threads:
        """
        super().__init__()
        self.model = None
        self.y_vars = None
        self.b_vars = None
        self.continuous = True
        self.n_threads = 1
        self.priority_sign = -1
        self.constraints = None
        self.n_chosen = None
        self.weights = None
        self.n_cnst_func = n_cnst_func

    def get_name(self):
        if self.constraints is None:
            return f"{type(self).__name__}_sad_{self.n_cnst_func}"
        else:
            return f"{type(self).__name__}_{self.n_chosen}_{len(self.constraints)}"

    def reset(self, matrix):
        self._times = {"model_preparation_time": 0, "optimization_time": 0}
        self.matrix = matrix
        model_time = time.time()
        self.model, self.y_vars, self.b_vars = StaticUtil_b.make_Gurobi_model(
            matrix, continuous=True, add_constraints=False)
        self.constraints = StaticUtil_b.get_constraints(matrix.shape[0], matrix.shape[1], self.y_vars, self.b_vars)
        if self.n_cnst_func is not None:
            self.n_chosen = self.n_cnst_func(self.matrix.shape[0], self.matrix.shape[1])
            self.n_chosen = int(min(len(self.constraints), self.n_chosen))
            sample = np.random.choice(len(self.constraints), self.n_chosen, replace = False)
        else:
            self.n_chosen = len(self.constraints)
            sample = range(self.n_chosen)
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

class StaticUtil_b():
    def __init__(self):
        pass

    @staticmethod
    def get_constraints(num_cells, num_mutations, Y, B):
        ret = []

        for i in range(num_cells):
            for p in range(num_mutations):
                for q in range(num_mutations):
                    if p!=q:
                        ret.append(Constraint(type="B11", row = i, col1=p, col2=q,
                                              exp=Y[i, p] + Y[i, q] - B[p, q, 1, 1] <= 1))
                        ret.append(Constraint(type="B01", row = i, col1=p, col2=q,
                                              exp=-Y[i, p] + Y[i, q] - B[p, q, 0, 1] <= 0))
                        ret.append(Constraint(type="B10", row = i, col1=p, col2=q,
                                              exp=Y[i, p] - Y[i, q] - B[p, q, 1, 0] <= 0))

        for p in range(num_mutations):
            for q in range(num_mutations):
                if p!= q:
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
    n, m = 5, 5
    x = np.random.randint(2, size=(n, m))
    # x = I6
    # x = I_small
    delta = sp.lil_matrix(x.shape)

    # xx = StaticUtil.get_constraints(n, m)
    # print(len(xx))
    # exit(0)
    funcs = [
        lambda n ,m: 200*n*m,
        lambda n ,m: 9*(3*n**1.05+1)*m**0.9,
        lambda n ,m: 10*n*m**1.5,
        lambda n ,m: 300*(n+m**2),
        lambda n, m: (3*n+1) * m *(m-1)
    ]
    for ind, f in enumerate(funcs):
        subsample_bounding_b = SubsampleLPBounding_b(f)
        subsample_bounding_b.reset(x)
        print(subsample_bounding_b.get_name())
        print(ind, subsample_bounding_b.get_bound(delta), subsample_bounding_b.get_times())
    exit(0)

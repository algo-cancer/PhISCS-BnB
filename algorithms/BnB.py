# from boundings.LP import *
from boundings.two_sat import *
# from boundings.LP_APX_b import *
from boundings.MWM import *
from utils.const import *
from utils.interfaces import *
from utils.util import *


def bnb_solve(matrix, bounding_algorithm, time_limit=3600, na_value=None):
    problem1 = BnB(matrix, bounding_algorithm, na_value=na_value)
    solver = pybnb.solver.Solver()
    results1 = solver.solve(problem1, queue_strategy="custom", log=None, time_limit=time_limit)
    if results1.solution_status != "unknown":
        returned_delta = results1.best_node.state[0]
        returned_delta_na = results1.best_node.state[-1]
        returned_matrix = get_effective_matrix(matrix, returned_delta, returned_delta_na, change_na_to_0=True)
    else:
        returned_matrix = np.zeros((1,1))
    print("results1.nodes:  ", results1.nodes)
    return returned_matrix, results1.termination_condition


class BnB(pybnb.Problem):
    """
    - Accept Bounding algorithm with the interface
    - uses gusfield if the bounding does not provide next p,q
    - only delta is getting copied
    """

    def __init__(self, I, boundingAlg: BoundingAlgAbstract, na_value=None):
        """
        :param I:
        :param boundingAlg:
        :param checkBounding:
        """
        self.na_value = na_value
        self.has_na = np.any(I == self.na_value)
        self.I = I
        self.delta = sp.lil_matrix(I.shape, dtype=np.int8)  # this can be coo_matrix too
        self.boundingAlg = boundingAlg
        self.delta_na = None
        if self.has_na:
            assert boundingAlg.na_support, "Input has N/A coordinates but bounding algorithm doesn't support it."
            self.delta_na = sp.lil_matrix(I.shape, dtype=np.int8)  # the coordinates with na that are decided to be 1
        self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        self.boundingAlg.reset(I)
        self.node_to_add = self.boundingAlg.get_init_node()
        self.bound_value = self.boundingAlg.get_bound(self.delta)

    # noinspection PyMethodMayBeStatic
    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.delta.count_nonzero()
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        # print(self.bound_value)
        return self.bound_value

    def save_state(self, node):
        node.state = (self.delta, self.icf, self.colPair, self.bound_value, self.boundingAlg.get_state(), self.delta_na)

    def load_state(self, node):
        self.delta, self.icf, self.colPair, self.bound_value, boundingAlgState, self.delta_na = node.state
        self.boundingAlg.set_state(boundingAlgState)

    def get_current_matrix(self):
        return get_effective_matrix(self.I, self.delta, self.delta_na)

    def branch(self):
        # print_line()
        if self.icf:  # Once a node is icf by flipping more than here, the objective is not going to get better
            return

        need_for_new_nodes = True
        if self.node_to_add is not None:
            newnode = self.node_to_add
            self.node_to_add = None

            if newnode.state[0].count_nonzero() == self.bound_value:  # current_obj == lb => no need to explore
                need_for_new_nodes = False
            assert newnode.queue_priority is not None, "Right before adding a node its priority in the queue is not set!"
            yield newnode
        if need_for_new_nodes:
            p, q = self.colPair
            nf01 = None
            current_matrix = self.get_current_matrix()
            for col, colp in [(q, p), (p, q)]:
                node = pybnb.Node()
                nodedelta = copy.deepcopy(self.delta)
                node_na_delta = copy.deepcopy(self.delta_na)
                col1 = np.array(current_matrix[:, col], dtype = np.int8).reshape(-1)
                col2 = np.array(current_matrix[:, colp], dtype = np.int8).reshape(-1)
                rows01 = np.nonzero(np.logical_and(col1 == 0, col2 == 1))[0]
                rows21 = np.nonzero(np.logical_and(col1 == self.na_value, col2 == 1))[0]
                if len(rows01) + len(rows21) == 0: # nothing has changed! Dont add new node
                    continue
                nodedelta[rows01, col] = 1
                nf01 = nodedelta.count_nonzero()
                if self.has_na:
                    node_na_delta[rows21, col] = 1
                    new_bound = self.boundingAlg.get_bound(nodedelta, node_na_delta)
                else:
                    new_bound = self.boundingAlg.get_bound(nodedelta)

                node_icf, nodecol_pair = None, None
                extra_info = self.boundingAlg.get_extra_info()

                if extra_info is not None:
                    if "icf" in extra_info:
                        node_icf = extra_info["icf"]
                    if "one_pair_of_columns" in extra_info:
                        nodecol_pair = extra_info["one_pair_of_columns"]
                if node_icf is None:
                    x = get_effective_matrix(self.I, nodedelta, node_na_delta)
                    node_icf, nodecol_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(x)

                node_bound_value = max(self.bound_value, new_bound)
                node.state = (nodedelta, node_icf, nodecol_pair, node_bound_value, self.boundingAlg.get_state(), node_na_delta)
                node.queue_priority = self.boundingAlg.get_priority(
                    till_here = nf01 - len(rows01),
                    this_step = len(rows01),
                    after_here = new_bound - nf01,
                    icf = node_icf)
                assert node.queue_priority is not None, "Right before adding a node its priority in the queue is not set!"
                yield node

    # def notify_new_best_node(self,
    #                          node,
    #                          current):
    #     print("-----------------------------------")
    #     next(plg)
    #     print(node)
    #     print("changing version")
    #     print("-----------------------------------")
    #     # self.version = 1




if __name__ == "__main__":
    time_limit = 60000
    x = np.array([[0, 1, 0, 1, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 1, 0, 1, 0],
       [1, 0, 1, 0, 1]])

    # n, m = 6, 6
    # x = np.random.randint(2, size=(n, m))
    # filename = "../../../PhISCS_BnB/Data/real/Chi-Ping.SC"
    # x = read_matrix_from_file(filename)
    # x = x[:, :500]
    print(x.shape)
    # print(repr(x))
    # opt = myPhISCS_I(x)
    # print("opt_I=", opt)
    # opt = myPhISCS_B(x)
    # print("opt_B=", opt)

    # setting = [False,]*5
    queue_strategy = "custom"
    # bnd = TwoSatBounding()
    bnd = DynamicMWMBounding(ascending_order=False)
    # bnd = DynamicMWMBounding(ascending_order=True)
    # bnd = RandomPartitioning(ascending_order=False)
    # bnd = RandomPartitioning(ascending_order=True)
    problem1 = BnB(x, bnd)
    solver = pybnb.solver.Solver()
    print("start solving...")
    results1 = solver.solve(problem1, queue_strategy=queue_strategy, log=None, time_limit=time_limit)
    print(results1)
    delta = results1.best_node.state[0]
    delta_na = results1.best_node.state[-1]
    current_matrix = get_effective_matrix(x, delta, delta_na)
    print("opt =", results1.objective)
    print("delta=", np.sum(delta))
    print("delta_na=", np.sum(delta_na))

    exit(0)


    optimTime_I = time.time()
    optim = myPhISCS_I(x)
    optimTime_I = time.time() - optimTime_I
    print("Optimal answer (I):", optim)
    print("Optimal time   (I):", optimTime_I)
    optimTime_B = time.time()
    optim = myPhISCS_B(x)
    optimTime_B = time.time() - optimTime_B
    print("Optimal answer (B):", optim)
    print("Optimal time   (B):", optimTime_B)

    if optim > 25:
        exit(0)

    boundings = [
        # EmptyBoundingAlg(),
        # (NaiveBounding(), 'fifo'), # The time measures of First one is not trusted for cache issues
        # (NaiveBounding(), 'depth'),
        # (NaiveBounding(), 'custom'),
        # (RandomPartitioning(ascendingOrder=True), 'custom'),
        # (SemiDynamicLPBounding(), 'fifo'),
        # (SemiDynamicLPBounding(), 'depth'),
        # (StaticILPBounding(), 'custom'),
        # (StaticILPBounding(), 'custom'),
        # (SemiDynamicLPBounding(), 'custom'),
        # (SemiDynamicLPBounding(), 'custom'),
        # (SemiDynamicLPBounding(), 'custom'),
        (SemiDynamicLPBounding(ratio=None, continuous=True), "custom"),
        # (SemiDynamicLPBounding(ratio=None, continuous = False), 'custom'),
        # (StaticLPBounding(), 'fifo'),
        # (StaticLPBounding(), 'custom'),
        # (StaticLPBounding(), 'custom'),
        # (StaticLPBounding(), 'custom'),
        # (StaticLPBounding(), 'depth'),
        (DynamicMWMBounding(), "custom"),
        # (DynamicMWMBounding(), 'custom'),
        # (DynamicMWMBounding(), 'custom'),
        # (DynamicMWMBounding(), 'fifo'),
        # (DynamicMWMBounding(), 'depth'),
        # (DynamicMWMBounding(), 'custom'),
        # (StaticMWMBounding(), 'custom'),
        # (StaticMWMBounding(), 'custom'),
        # (StaticMWMBounding(), 'custom'),
        # (StaticMWMBounding(), 'depth')
    ]

    for boundFunc, queue_strategy in boundings:
        time1 = time.time()
        problem1 = BnB(x, boundFunc, False)
        solver = pybnb.solver.Solver()
        results1 = solver.solve(problem1, queue_strategy=queue_strategy, log=None, time_limit=timeLimit)
        # results1 = solver.solve(problem1,  queue_strategy = queue_strategy,)
        time1 = time.time() - time1
        delta = results1.best_node.state[0]
        nf1 = delta.count_nonzero()
        print(nf1, str(time1)[:5], results1.nodes, boundFunc.get_name(), queue_strategy, flush=True)


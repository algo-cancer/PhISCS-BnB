from Utils.const import *
from Utils.interfaces import *
from Utils.util import *
from Boundings.LP import *
from Boundings.two_sat import *
from Boundings.LP_APX_b import *
from Boundings.MWM import *


class BnB(pybnb.Problem):
    """
    - Accept Bounding algorithm with the interface
    - uses gusfield if the bounding does not provide next p,q
    - only delta is getting copied
    """

    def __init__(self, I, boundingAlg: BoundingAlgAbstract, checkBounding=False, version=0):
        """
        :param I:
        :param boundingAlg:
        :param checkBounding:
        :param version: An integer 0: our normal alg in Nov 11th
                                   1: change the zeros of the whole column at the same time.
        """
        self.na_value = 2 # todo put this somewhere in config
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
        self.boundVal = self.boundingAlg.get_bound(self.delta)
        self.checkBounding = checkBounding
        self.version = version

    # def getNFlips(self):
    #     return self.delta.count_nonzero()

    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.delta.count_nonzero()
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        if self.checkBounding:  # Debugging here
            ll = myPhISCS_I(np.asarray(self.I + self.delta))
            if ll + self.delta.count_nonzero() < self.boundVal:
                print(ll, self.getNFlips(), self.boundVal)
                print(" ============== ")
                print(repr(self.I))
                print(repr(self.delta.todense()))
                print(" ============== ")
                ss = SemiDynamicLPBounding()
                ss.reset(self.I)
                print(f"np.allclose(self.boundingAlg.matrix, self.I)={np.allclose(self.boundingAlg.matrix, self.I)}")
                print(f"len(self.boundingAlg.model.getConstrs()) = {len(self.boundingAlg.model.getConstrs())}")
                print(f"len(ss.model.getConstrs()) = {len(ss.model.getConstrs())}")
                thisAnswer = ss.get_bound(self.delta)
                print(f"np.allclose(self.boundingAlg.matrix, self.I)={np.allclose(self.boundingAlg.matrix, self.I)}")
                print(f"len(self.boundingAlg.model.getConstrs()) = {len(self.boundingAlg.model.getConstrs())}")
                print(f"len(ss.model.getConstrs()) = {len(ss.model.getConstrs())}")

                print(f"{thisAnswer} vs {self.boundingAlg.get_bound(self.delta)} vs {self.boundVal}")
                print(f"{thisAnswer} vs {self.boundingAlg.get_bound(self.delta)} vs {self.boundVal}")

                exit(0)
        return self.boundVal

    def save_state(self, node):
        node.state = (self.delta, self.icf, self.colPair, self.boundVal, self.boundingAlg.get_state(), self.delta_na)

    def load_state(self, node):
        self.delta, self.icf, self.colPair, self.boundVal, boundingAlgState, self.delta_na = node.state
        self.boundingAlg.set_state(boundingAlgState)

    def getCurrentMatrix(self):
        return get_effective_matrix(self.I, self.delta, self.delta_na)

    def branch(self):
        if self.icf:  # by fliping more than here, the objective is not going to get better
            return
        if self.node_to_add is not None:
            newnode = self.node_to_add
            self.node_to_add = None
            yield newnode
        p, q = self.colPair
        if self.version == 0:
            assert not self.has_na, "This version does not support N/A yet"
            p, q, oneone, zeroone, onezero = get_a_coflict(self.getCurrentMatrix(), p, q)
            # nodes = []
            nf = self.getNFlips() + 1
            for a, b in [(onezero, q), (zeroone, p)]:
                node = pybnb.Node()
                nodedelta = copy.deepcopy(self.delta)
                nodedelta[a, b] = 1

                newBound = self.boundingAlg.get_bound(nodedelta)

                nodeicf, nodecolPair = None, None
                extraInfo = self.boundingAlg.get_extra_info()
                if extraInfo is not None:
                    if "icf" in extraInfo:
                        nodeicf = extraInfo["icf"]
                    if "one_pair_of_columns" in extraInfo:
                        nodecolPair = extraInfo["one_pair_of_columns"]
                if nodeicf is None:
                    nodeicf, nodecolPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I + nodedelta)

                nodeboundVal = max(self.boundVal, newBound)
                node.state = (nodedelta, nodeicf, nodecolPair, nodeboundVal, self.boundingAlg.get_state(), None)
                node.queue_priority = self.boundingAlg.get_priority(
                    till_here = nf - 1,
                    this_step = 1,
                    after_here = newBound - nf,
                    icf = nodeicf)
                yield node
        elif self.version == 1:
            nf01 = None
            current_matrix = self.getCurrentMatrix()
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
                    newBound = self.boundingAlg.get_bound(nodedelta, node_na_delta)
                else:
                    newBound = self.boundingAlg.get_bound(nodedelta)

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

                nodeboundVal = max(self.boundVal, newBound)
                node.state = (nodedelta, node_icf, nodecol_pair, nodeboundVal, self.boundingAlg.get_state(), node_na_delta)
                node.queue_priority = self.boundingAlg.get_priority(
                    till_here = nf01 - len(rows01),
                    this_step = len(rows01),
                    after_here = newBound - nf01,
                    icf = node_icf)
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
    time_limit = 60
    # n, m = 6, 6
    # x = np.random.randint(2, size=(n, m))
    x = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")
    x[0, 0] = 2
    print(repr(x))
    opt = myPhISCS_I(x)
    print("opt_I=", opt)
    opt = myPhISCS_B(x)
    print("opt_B=", opt)

    queue_strategy = "custom"
    for i in range(1, 2):
        bnd = two_sat(priority_version=1, formulation_version=0, formulation_threshold=0)
        problem1 = BnB(x, bnd, False, i)
        solver = pybnb.solver.Solver()
        results1 = solver.solve(problem1, queue_strategy=queue_strategy, log=None, time_limit=time_limit)
        print(results1)
        delta = results1.best_node.state[0]
        delta_na = results1.best_node.state[-1]
        print("opt =", opt, results1.objective)
        print("delta=", delta)
        print("delta_na=", delta_na)
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


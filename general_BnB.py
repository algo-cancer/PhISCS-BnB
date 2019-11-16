from Utils.const import *
from Utils.interfaces import *
from Utils.util import *
from Boundings.LP import *
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
        self.I = I
        self.delta = sp.lil_matrix(I.shape, dtype=np.int8)  # this can be coo_matrix too
        self.icf, self.colPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I)
        self.boundingAlg = boundingAlg
        self.boundingAlg.reset(I)
        self.boundVal = self.boundingAlg.get_bound(self.delta)
        self.checkBounding = checkBounding
        self.version = version

    def getNFlips(self):
        return self.delta.count_nonzero()

    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.getNFlips()
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
        node.state = (self.delta, self.icf, self.colPair, self.boundVal, self.boundingAlg.get_state())

    def load_state(self, node):
        self.delta, self.icf, self.colPair, self.boundVal, boundingAlgState = node.state
        self.boundingAlg.set_state(boundingAlgState)

    def getCurrentMatrix(self):
        return self.I + self.delta

    def branch(self):
        if self.icf:  # by fliping more the objective is not going to get better
            return
        p, q = self.colPair
        if self.version == 0:
            p, q, oneone, zeroone, onezero = get_a_coflict(self.getCurrentMatrix(), p, q)
            # nodes = []
            nf = self.getNFlips() + 1
            for a, b in [(onezero, q), (zeroone, p)]:
                # print(f"{(a,b)} is made!")
                node = pybnb.Node()
                nodedelta = copy.deepcopy(self.delta)
                nodedelta[a, b] = 1

                newBound = self.boundingAlg.get_bound(nodedelta)

                nodeicf, nodecolPair = None, None
                extraInfo = self.boundingAlg.get_extra_info()
                # print(extraInfo)
                if extraInfo is not None:
                    if "icf" in extraInfo:
                        nodeicf = extraInfo["icf"]
                    if "one_pair_of_columns" in extraInfo:
                        nodecolPair = extraInfo["one_pair_of_columns"]
                if nodeicf is None or nodecolPair is None:
                    # print("run gusfield")
                    nodeicf, nodecolPair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I + nodedelta)

                nodeboundVal = max(self.boundVal, newBound)
                node.state = (nodedelta, nodeicf, nodecolPair, nodeboundVal, self.boundingAlg.get_state())
                # node.queue_priority = - ( newBound - nf)
                node.queue_priority = self.boundingAlg.get_priority(newBound - nf, nodeicf)
                # node.queue_priority =  self.boundingAlg.getPriority(nodeboundVal)
                yield node
        elif self.version == 1:
            nf = None
            current_matrix = self.getCurrentMatrix()
            for col, colp in [(q, p), (p, q)]:
                node = pybnb.Node()
                nodedelta = copy.deepcopy(self.delta)
                col1 = np.array(current_matrix[:, col], dtype = np.bool).reshape(-1)
                col2 = np.array(current_matrix[:, colp], dtype = np.bool).reshape(-1)
                rows = (col1 < col2).nonzero() # 0, 1 s

                nodedelta[rows, col] = 1

                nf = nodedelta.count_nonzero()
                if nf == self.getNFlips():
                    continue
                newBound = self.boundingAlg.get_bound(nodedelta)

                node_icf, nodecol_pair = None, None
                extra_info = self.boundingAlg.get_extra_info()
                # print(extra_info)
                if extra_info is not None:
                    if "icf" in extra_info:
                        nodeicf = extra_info["icf"]
                    if "one_pair_of_columns" in extra_info:
                        nodecol_pair = extra_info["one_pair_of_columns"]
                if node_icf is None:
                    # print("run gusfield")
                    node_icf, nodecol_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I + nodedelta)

                nodeboundVal = max(self.boundVal, newBound)
                node.state = (nodedelta, node_icf, nodecol_pair, nodeboundVal, self.boundingAlg.get_state())
                node.queue_priority = self.boundingAlg.get_priority(newBound - nf, node_icf)
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



# def ErfanBnBSolver(x):
#   problem = BnB(x, EmptyBoundingAlg())
#   results = pybnb.solve(problem) #, log=None)
#   ans = results.best_node.state[0]
#   return ans

if __name__ == "__main__":
    time_limit = 60

    # n, m = 30, 4
    # n, m = 5, 5
    # x = np.random.randint(2, size=(n, m))
    x = read_matrix_from_file()
    # x = np.array(
    #     [
    #         [0, 1, 0, 0,],
    #         [0, 1, 1, 0,],
    #         [1, 0, 0, 1,],
    #         [1, 1, 1, 1,],
    #         # [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         # [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    #         # [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #         # [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    #         # [0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    #         # [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    #     ]
    # )
    # x = np.array(
    #     [
    #         [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    #         [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    #         [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    #         [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    #         [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #         [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    #         [0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    #     ]
    # )

    # x = np.array([[0, 1, 1, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    # x = np.array(
    #     [
    #         [1, 0],
    #         [1, 0],
    #         [1, 0],
    #         [0, 1],
    #         [0, 1],
    #         [0, 1],
    #         [1, 1]
    #     ]
    # )
    print(repr(x))
    queue_strategy = "custom"
    time_limit = 300
    plg= print_line_iter()
    next(plg)
    for i in range(2):
        next(plg)
        problem1 = BnB(x, SubsampleLPBounding_b(lambda n, m: int(0.6 * (3 * n + 1) * m ** 1.8)), False, i)
        # problem1 = BnB(x, SemiDynamicLPBounding(), False, i)
        # problem1 = BnB(x, DynamicMWMBounding(), False, i)
        solver = pybnb.solver.Solver()
        results1 = solver.solve(problem1, queue_strategy=queue_strategy, log=None, time_limit=time_limit)
        print(results1)
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


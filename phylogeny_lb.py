import multiprocessing
from Utils.const import *
from Utils.util import *
from ortools.linear_solver import pywraplp


def lb_phiscs_b(D, a, b):
    # def get_partition_random(D, n_group_members=5):
    #     d = int(D.shape[1]/n_group_members)
    #     partitions_id = np.random.choice(range(D.shape[1]), size=(d, n_group_members), replace=False)
    #     return partitions_id
    def blockshaped(arr, nrows, ncols):
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))
    
    ## 1)
    lb = 0
    for block in blockshaped(D, D.shape[0], 5):
        solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1), c_time = PhISCS_B(block)
        lb += flips_0_1
    
    ## 2)
    # lb = 0
    # blocks = blockshaped(D, D.shape[0], 5)
    # with Pool(processes=len(blocks)) as pool:
    #     result = pool.map(PhISCS_B, blocks)
    # for x in result:
    #     lb += x[1][0]

    ## 3)
    # blocks = blockshaped(D, D.shape[0], 5)
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # jobs = []
    # for i in range(len(blocks)):
    #     p = multiprocessing.Process(target=PhISCS_B, args=(blocks[i],i,return_dict))
    #     jobs.append(p)
    #     p.start()
    
    # for proc in jobs:
    #     proc.join()
    # lb = sum(return_dict.values())

    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return lb, {}, best_pair_qp, icf


def lb_greedy(D, a, b):
    def get_important_pair_of_columns_in_conflict(D):
        important_columns = defaultdict(lambda: 0)
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                oneone = 0
                zeroone = 0
                onezero = 0
                for r in range(D.shape[0]):
                    if D[r,p] == 1 and D[r,q] == 1:
                        oneone += 1
                    if D[r,p] == 0 and D[r,q] == 1:
                        zeroone += 1
                    if D[r,p] == 1 and D[r,q] == 0:
                        onezero += 1
                if oneone > 0:
                    important_columns[(p,q)] += min(zeroone, onezero)
        return important_columns
    
    ipofic = get_important_pair_of_columns_in_conflict(D)
    sorted_ipofic = sorted(ipofic.items(), key=operator.itemgetter(1), reverse=True)
    pairs = [sorted_ipofic[0][0]]
    elements = [sorted_ipofic[0][0][0], sorted_ipofic[0][0][1]]
    sorted_ipofic.remove(sorted_ipofic[0])
    for x in sorted_ipofic[:]:
        notFound = True
        for y in x[0]:
            if y in elements:
                sorted_ipofic.remove(x)
                notFound = False
                break
        if notFound:
            pairs.append(x[0])
            elements.append(x[0][0])
            elements.append(x[0][1])
    lb = 0
    for x in pairs:
        lb += ipofic[x]
    
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return lb, {}, best_pair_qp, icf
    

def lb_random(D, a, b):
    def get_partition_random(D):
        d = int(D.shape[1]/2)
        partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
        return partitions_id
    
    def calc_min0110_for_one_pair_of_columns(D, p, q):
        foundOneOne = False
        numberOfZeroOne = 0
        numberOfOneZero = 0
        for r in range(D.shape[0]):
            if D[r,p] == 1 and D[r,q] == 1:
                foundOneOne = True
            if D[r,p] == 0 and D[r,q] == 1:
                numberOfZeroOne += 1
            if D[r,p] == 1 and D[r,q] == 0:
                numberOfOneZero += 1
        if foundOneOne:
            return min(numberOfZeroOne, numberOfOneZero)
        else:
            return 0

    lb = 0
    for x in get_partition_random(D):
        lb += calc_min0110_for_one_pair_of_columns(D, x[0], x[1])
    
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return lb, {}, best_pair_qp, icf


def lb_openwbo(D, a, b):
    solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1), c_time = PhISCS_B(D)
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return flips_0_1, {}, best_pair_qp, icf


def lb_gurobi(D, a, b):
    solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1), c_time = PhISCS_I(D, beta=0.9, alpha=0.00000001)
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(D)
    return flips_0_1, {}, best_pair_qp, icf


def lb_lp_ortools(I, a, b):
    a = time.time()
    model = pywraplp.Solver('LP_ORTOOLS', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    numCells = I.shape[0]
    numMutations = I.shape[1]
    Y = {}
    numOne = 0
    for c in range(numCells):
        for m in range(numMutations):
            if I[c, m] == 0:
                Y[c, m] = model.NumVar(0, 1, 'Y({0},{1})'.format(c, m))
            elif I[c, m] == 1:
                numOne += 1
                Y[c, m] = 1
    B = {}
    for p in range(numMutations):
        for q in range(numMutations):
            B[p, q, 1, 1] = model.NumVar(0, 1, 'B[{0},{1},1,1]'.format(p, q))
            B[p, q, 1, 0] = model.NumVar(0, 1, 'B[{0},{1},1,0]'.format(p, q))
            B[p, q, 0, 1] = model.NumVar(0, 1, 'B[{0},{1},0,1]'.format(p, q))
    for i in range(numCells):
        for p in range(numMutations):
            for q in range(numMutations):
                model.Add(Y[i,p] + Y[i,q] - B[p,q,1,1] <= 1)
                model.Add(-Y[i,p] + Y[i,q] - B[p,q,0,1] <= 0)
                model.Add(Y[i,p] - Y[i,q] - B[p,q,1,0] <= 0)
    for p in range(numMutations):
        for q in range(numMutations):
            model.Add(B[p,q,0,1] + B[p,q,1,0] + B[p,q,1,1] <= 2)

    objective = sum([Y[c, m] for c in range(numCells) for m in range(numMutations)])
    model.Minimize(objective)
    b = time.time()
    result_status = model.Solve()
    c = time.time()
    optimal_solution = model.Objective().Value()
    lb = np.int(np.ceil(optimal_solution)) - numOne

    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
    d = time.time()
    t1 = b-a
    t2 = c-b
    t3 = d-c
    return lb, {}, best_pair_qp, icf, t1, t2, t3


def lb_lp_gurobi(I, flips, previous_model):
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout
    
    with HiddenPrints():
        ta = time.time()
        if flips == None:
            model = Model('LP_GUROBI')
            model.Params.LogFile = ''
            model.Params.Threads = 1
            # model.setParam('TimeLimit', 0.005)
            numCells = I.shape[0]
            numMutations = I.shape[1]
            Y = {}
            for c in range(numCells):
                for m in range(numMutations):
                    if I[c, m] == 0:
                        Y[c, m] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=1, name='Y({0},{1})'.format(c, m))
                    elif I[c, m] == 1:
                        Y[c, m] = 1
            B = {}
            for p in range(numMutations):
                for q in range(numMutations):
                    B[p, q, 1, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0, name='B[{0},{1},1,1]'.format(p, q))
                    B[p, q, 1, 0] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0, name='B[{0},{1},1,0]'.format(p, q))
                    B[p, q, 0, 1] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, obj=0, name='B[{0},{1},0,1]'.format(p, q))
            for i in range(numCells):
                for p in range(numMutations):
                    for q in range(numMutations):
                        model.addConstr(Y[i,p] + Y[i,q] - B[p,q,1,1] <= 1)
                        model.addConstr(-Y[i,p] + Y[i,q] - B[p,q,0,1] <= 0)
                        model.addConstr(Y[i,p] - Y[i,q] - B[p,q,1,0] <= 0)
            for p in range(numMutations):
                for q in range(numMutations):
                    model.addConstr(B[p,q,0,1] + B[p,q,1,0] + B[p,q,1,1] <= 2)
            model.Params.ModelSense = GRB.MINIMIZE
            tb = time.time()
            model.optimize()
            lb = np.int(np.ceil(model.objVal))
            tc = time.time()
        else:
            model = previous_model
            new_constrs = (model.getVarByName('Y({0},{1})'.format(i,j)) == 1 for i,j in flips)
            new_constrs_returned = model.addConstrs(new_constrs)
            model.update()
            tb = time.time()
            model.optimize()
            lb = np.int(np.ceil(model.objVal)) - len(flips) - 1
            for cnstr in new_constrs_returned.values():
                model.remove(cnstr)
            model.update()
            tc = time.time()
        
    icf, best_pair_qp = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I)
    td = time.time()
    t1 = tb-ta
    t2 = tc-tb
    t3 = td-tc
    return lb, model, best_pair_qp, icf, t1, t2, t3


def lb_max_weight_matching(D, changed_column, previous_G):
    if changed_column == None:
        G = nx.Graph()
    else:
        G = previous_G

    def calc_min0110_for_one_pair_of_columns(p, q, G):
        foundOneOne = False
        numberOfZeroOne = 0
        numberOfOneZero = 0
        for r in range(D.shape[0]):
            if D[r,p] == 1 and D[r,q] == 1:
                foundOneOne = True
            if D[r,p] == 0 and D[r,q] == 1:
                numberOfZeroOne += 1
            if D[r,p] == 1 and D[r,q] == 0:
                numberOfOneZero += 1
        if foundOneOne:
            G.add_edge(p, q, weight=min(numberOfZeroOne, numberOfOneZero))
        else:
            G.add_edge(p, q, weight=0)

    ta = time.time()
    if changed_column == None:
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                calc_min0110_for_one_pair_of_columns(p, q, G)
    else:
        q = changed_column
        for p in range(D.shape[1]):
            if p < q:
                calc_min0110_for_one_pair_of_columns(p, q, G)
            elif q < p:
                calc_min0110_for_one_pair_of_columns(q, p, G)

    tb = time.time()
    best_pairing = nx.max_weight_matching(G)
    tc = time.time()
    # print(best_pairing)
    best_pair_qp, best_pair_w = (None, None), np.inf
    # for (u, v, wt) in G.edges.data('weight'):
    #     if wt > best_pair_w:
    #         best_pair_qp = (u, v)
    #         best_pair_w = wt
        # print(u, v, wt)
    lb = 0
    for a, b in best_pairing:
        # print(a,b,G[a][b]["weight"])
        if G[a][b]["weight"] < best_pair_w:
            best_pair_w = G[a][b]["weight"]
            best_pair_qp = (a, b)
        lb += G[a][b]["weight"]
    if lb == 0:
        icf = True
    else:
        icf = False
    td = time.time()
    t1 = tb-ta
    t2 = tc-tb
    t3 = td-tc
    return lb, G, best_pair_qp, icf, t1, t2, t3

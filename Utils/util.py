from Utils.const import *



def print_line(depth=1, shift=1):
    """A debugging tool!  """
    stack = inspect.stack()
    for i in range(shift, min(len(stack), depth + shift)):
        info = stack[i]
        for j in range(i - 1):
            print("\t", end="")
        print(f"Line {info.lineno} in {info.filename}, Function: {info.function}")

def print_line_iter(depth=1):
    """A debugging tool!  """
    last_time = 0
    while(True):
        last_time = time.time() - last_time
        print(last_time, end="")
        print_line(depth=depth, shift=2)
        print(flush=True)
        last_time = time.time()
        yield
        # return

def read_matrix_from_file(
        file_name = "simNo_2-s_4-m_50-n_50-k_50.SC.noisy",
        args = {"simNo":2, "s": 4, "m": 50, "n":50, "k":50, "kind": "noisy"}, folder_path = noisy_folder_path):
    assert file_name is not None or args is not None, "give an input"

    if file_name is None:
        file_name = f"simNo_{args['simNo']}-s_{args['s']}-m_{args['m']}-n_{args['n']}.SC.{args['kind']}"
        if args['kind'] == "noisy":
            folder_path = noisy_folder_path
        elif args['kind'] == "ground":
            folder_path = simulation_folder_path
    file = folder_path + file_name
    df_sim = pd.read_csv(file, delimiter="\t", index_col=0)
    return df_sim.values


def timed_run(func, args, time_limit=1):
    def internal_func(shared_dict):
        args_to_pass = dict()
        args_needed = inspect.getfullargspec(func).args
        for arg in args_needed:
            args_to_pass[arg] = shared_dict["input"][arg]

        runtime = time.time()
        shared_dict["output"] = func(**args_to_pass)
        runtime = time.time() - runtime
        shared_dict["runtime"] = runtime

    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    shared_dict["input"] = args
    p = multiprocessing.Process(target=internal_func, name="internal_func", args=(shared_dict,))
    p.start()
    p.join(time_limit)
    # If thread is active
    if p.is_alive():
        p.terminate()
        # p.join()
        shared_dict["output"] = None
        shared_dict["termination_condition"] = 'time_limit'
        shared_dict["runtime"] = time_limit
    else:
        shared_dict["termination_condition"] = 'success'
    return shared_dict


def get_matrix_hash(x):
    return hash(x.tostring()) % 10000000


def myPhISCS_B(x):
    solution, (f_0_1_b, f_1_0_b, f_2_0_b, f_2_1_b), cb_time = PhISCS_B(x, beta=0.97, alpha=0.00001)
    nf = len(np.where(np.logical_and(solution != x, x != 2))[0])
    return nf


def myPhISCS_I(x):
    ret = PhISCS_I(x, beta=0.99, alpha=0.00001)
    solution = ret[0]
    nf = len(np.where(np.logical_and(solution != x, x != 2))[0])
    print("solution=", solution)
    return nf


def get_a_coflict(D, p, q):
    # todo: oneone is not important you can get rid of
    oneone = None
    zeroone = None
    onezero = None
    for r in range(D.shape[0]):
        if D[r, p] == 1 and D[r, q] == 1:
            oneone = r
        if D[r, p] == 0 and D[r, q] == 1:
            zeroone = r
        if D[r, p] == 1 and D[r, q] == 0:
            onezero = r
        if oneone != None and zeroone != None and onezero != None:
            return (p, q, oneone, zeroone, onezero)
    return None


def is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I):
    def sort_bin(a):
        b = np.transpose(a)
        b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
        idx = np.argsort(b_view.ravel())[::-1]
        c = b[idx]
        return np.transpose(c), idx

    I = I.copy()
    I[I==2] = 0
    O, idx = sort_bin(I)
    # TODO: delete duplicate columns
    # print(O, '\n')
    Lij = np.zeros(O.shape, dtype=int)
    for i in range(O.shape[0]):
        maxK = 0
        for j in range(O.shape[1]):
            if O[i, j] == 1:
                Lij[i, j] = maxK
                maxK = j + 1
    # print(Lij, '\n')
    Lj = np.amax(Lij, axis=0)
    # print(Lj, '\n')
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            if O[i, j] == 1:
                if Lij[i, j] != Lj[j]:
                    return False, (idx[j], idx[Lj[j] - 1])
    return True, (None, None)


def is_conflict_free_farid(D):
    conflict_free = True
    for p in range(D.shape[1]):
        for q in range(p + 1, D.shape[1]):
            oneone = False
            zeroone = False
            onezero = False
            for r in range(D.shape[0]):
                if D[r][p] == 1 and D[r][q] == 1:
                    oneone = True
                if D[r][p] == 0 and D[r][q] == 1:
                    zeroone = True
                if D[r][p] == 1 and D[r][q] == 0:
                    onezero = True
            if oneone and zeroone and onezero:
                conflict_free = False
    return conflict_free


def get_lower_bound_new(noisy, partition_randomly=False):
    def get_important_pair_of_columns_in_conflict(D):
        important_columns = defaultdict(lambda: 0)
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                oneone = 0
                zeroone = 0
                onezero = 0
                for r in range(D.shape[0]):
                    if D[r, p] == 1 and D[r, q] == 1:
                        oneone += 1
                    if D[r, p] == 0 and D[r, q] == 1:
                        zeroone += 1
                    if D[r, p] == 1 and D[r, q] == 0:
                        onezero += 1
                ## greedy approach based on the number of conflicts in a pair of columns
                # if oneone*zeroone*onezero > 0:
                #     important_columns[(p,q)] += oneone*zeroone*onezero
                ## greedy approach based on the min number of 01 or 10 in a pair of columns
                if oneone > 0:
                    important_columns[(p, q)] += min(zeroone, onezero)
        return important_columns

    def get_partition_sophisticated(D):
        ipofic = get_important_pair_of_columns_in_conflict(D)
        if len(ipofic) == 0:
            return []
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
        # print(sorted_ipofic, pairs, elements)
        partitions = []
        for x in pairs:
            partitions.append(D[:, x])
        return partitions

    def get_partition_random(D):
        d = int(D.shape[1] / 2)
        partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
        partitions = []
        for x in partitions_id:
            partitions.append(D[:, x])
        return partitions

    def get_lower_bound_for_a_pair_of_columns(D):
        foundOneOne = False
        numberOfZeroOne = 0
        numberOfOneZero = 0
        for r in range(D.shape[0]):
            if D[r, 0] == 1 and D[r, 1] == 1:
                foundOneOne = True
            if D[r, 0] == 0 and D[r, 1] == 1:
                numberOfZeroOne += 1
            if D[r, 0] == 1 and D[r, 1] == 0:
                numberOfOneZero += 1
        if foundOneOne:
            if numberOfZeroOne * numberOfOneZero > 0:
                return min(numberOfZeroOne, numberOfOneZero)
        return 0

    LB = []
    if partition_randomly:
        partitions = get_partition_random(noisy)
    else:
        partitions = get_partition_sophisticated(noisy)
    for D in partitions:
        LB.append(get_lower_bound_for_a_pair_of_columns(D))
    return sum(LB)


def get_lower_bound(noisy, partition_randomly=False):
    def get_important_pair_of_columns_in_conflict(D):
        important_columns = defaultdict(lambda: 0)
        for p in range(D.shape[1]):
            for q in range(p + 1, D.shape[1]):
                oneone = 0
                zeroone = 0
                onezero = 0
                for r in range(D.shape[0]):
                    if D[r, p] == 1 and D[r, q] == 1:
                        oneone += 1
                    if D[r, p] == 0 and D[r, q] == 1:
                        zeroone += 1
                    if D[r, p] == 1 and D[r, q] == 0:
                        onezero += 1
                if oneone * zeroone * onezero > 0:
                    important_columns[(p, q)] += oneone * zeroone * onezero
        return important_columns

    def get_partition_sophisticated(D):
        ipofic = get_important_pair_of_columns_in_conflict(D)
        if len(ipofic) == 0:
            return []
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
        # print(sorted_ipofic, pairs, elements)
        partitions = []
        for x in pairs:
            partitions.append(D[:, x])
        return partitions

    def get_partition_random(D):
        d = int(D.shape[1] / 2)
        partitions_id = np.random.choice(range(D.shape[1]), size=(d, 2), replace=False)
        partitions = []
        for x in partitions_id:
            partitions.append(D[:, x])
        return partitions

    def get_lower_bound_for_a_pair_of_columns(D):
        foundOneOne = False
        numberOfZeroOne = 0
        numberOfOneZero = 0
        for r in range(D.shape[0]):
            if D[r, 0] == 1 and D[r, 1] == 1:
                foundOneOne = True
            if D[r, 0] == 0 and D[r, 1] == 1:
                numberOfZeroOne += 1
            if D[r, 0] == 1 and D[r, 1] == 0:
                numberOfOneZero += 1
        if foundOneOne:
            if numberOfZeroOne * numberOfOneZero > 0:
                return min(numberOfZeroOne, numberOfOneZero)
        return 0

    LB = []
    if partition_randomly:
        partitions = get_partition_random(noisy)
    else:
        partitions = get_partition_sophisticated(noisy)
    for D in partitions:
        LB.append(get_lower_bound_for_a_pair_of_columns(D))
    return sum(LB)


def make_noisy_by_k(data, k):
    data2 = data.copy()
    n, m = np.where(data2 == 1)
    assert k <= len(n), 'k is greater than the number of ones in the input matrix!'
    s = np.random.choice(len(n), k, replace=False)
    assert len(s) == k
    for i in s:
        assert data2[n[i], m[i]] == 1
        data2[n[i], m[i]] = 0
    return data2


def make_noisy_by_fn(data, fn, fp, na):
    def toss(p):
        return True if np.random.random() < p else False
    
    if fn > 1:
        fn = fn / np.count_nonzero(data == 0)
        if fn > 1:
            fn = 0.999
    
    n, m = data.shape
    data2 = -1 * np.ones(shape=(n, m)).astype(int)
    countFP = 0
    countFN = 0
    countNA = 0
    countOneZero = 0
    indexNA = []
    changedBefore = []
    for i in range(n):
        for j in range(m):
            indexNA.append([i, j])
            countOneZero = countOneZero + 1
    random.shuffle(indexNA)
    nas = math.ceil(countOneZero * na)
    for i in range(int(nas)):
        [a, b] = indexNA[i]
        changedBefore.append([a, b])
        data2[a][b] = 2
        countNA = countNA + 1
    for i in range(n):
        for j in range(m):
            if data2[i][j] != 2:
                if data[i][j] == 1:
                    if toss(fn):
                        data2[i][j] = 0
                        countFN = countFN + 1
                    else:
                        data2[i][j] = data[i][j]
                elif data[i][j] == 0:
                    if toss(fp):
                        data2[i][j] = 1
                        countFP = countFP + 1
                    else:
                        data2[i][j] = data[i][j]
    return data2, (countFN, countFP, countNA)

def get_data_by_ms(n, m, seed, fn, fp, na, ms_path=ms_path):
    def build_ground_by_ms(n, m, seed):
        command = "{ms} {n} 1 -s {m} -seeds 7369 217 {r} | tail -n {n}".format(ms=ms_path, n=n, m=m, r=seed)
        result = os.popen(command).read()
        data = np.empty((n, m), dtype=int)
        i = 0
        for s in result.split("\n"):
            j = 0
            for c in list(s):
                data[i, j] = int(c)
                j += 1
            i += 1
        return data

    ground = build_ground_by_ms(n, m, seed)
    if is_conflict_free_farid(ground):
        noisy, (countFN, countFP, countNA) = make_noisy_by_fn(ground, fn, fp, na)
        if not is_conflict_free_farid(noisy) or fn + fp + na == 0:
            return ground, noisy, (countFN, countFP, countNA)
        else:
            return get_data_by_ms(n, m, seed + 1, fn, fp, na, ms_path)
    else:
        print("********************** ERROR ********************")
        print("Ground from ms is not Conflict free!")
        return get_data_by_ms(n, m, seed + 1, fn, fp, na, ms_path)


def count_flips(I, sol_K, sol_Y):
    flips_0_1 = 0
    flips_1_0 = 0
    flips_2_0 = 0
    flips_2_1 = 0
    n, m = I.shape
    for i in range(n):
        for j in range(m):
            if sol_K[j] == 0:
                if I[i][j] == 0 and sol_Y[i][j] == 1:
                    flips_0_1 += 1
                elif I[i][j] == 1 and sol_Y[i][j] == 0:
                    flips_1_0 += 1
                elif I[i][j] == 2 and sol_Y[i][j] == 0:
                    flips_2_0 += 1
                elif I[i][j] == 2 and sol_Y[i][j] == 1:
                    flips_2_1 += 1
    return (flips_0_1, flips_1_0, flips_2_0, flips_2_1)


def PhISCS_I(I, beta=0.99, alpha=0.00001, time_limit = 3600):
    def nearestInt(x):
        return int(x+0.5)

    logb1ma, log1mba, log1ma, loga = \
        np.log(beta / (1 - alpha)), np.log((1 - beta) / alpha), np.log((1 - alpha)), np.log(alpha)

    # scale = 1
    # logb1ma, log1mba, log1ma, loga = scale * logb1ma, scale * log1mba, scale * log1ma, scale * loga
    if  - log1mba / logb1ma > 70:
        logb1ma = -1
        log1mba = 70
        # print("change")

    # print(beta, logb1ma, log1mba, log1ma, loga)
    numCells, numMutations = I.shape
    sol_Y = []
    model = Model('PhISCS_ILP')
    model.Params.LogFile = ''
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    # model.setParam('TimeLimit', 10*60)

    Y = {}
    for c in range(numCells):
        for m in range(numMutations):
                Y[c, m] = model.addVar(vtype=GRB.BINARY, name='Y({0},{1})'.format(c, m))
    B = {}
    for p in range(numMutations):
        for q in range(numMutations):
            B[p, q, 1, 1] = model.addVar(vtype=GRB.BINARY, obj=0, name='B[{0},{1},1,1]'.format(p, q))
            B[p, q, 1, 0] = model.addVar(vtype=GRB.BINARY, obj=0, name='B[{0},{1},1,0]'.format(p, q))
            B[p, q, 0, 1] = model.addVar(vtype=GRB.BINARY, obj=0, name='B[{0},{1},0,1]'.format(p, q))
    model.update()
    for i in range(numCells):
        for p in range(numMutations):
            for q in range(numMutations):
                model.addConstr(Y[i,p] + Y[i,q] - B[p,q,1,1] <= 1)
                model.addConstr(-Y[i,p] + Y[i,q] - B[p,q,0,1] <= 0)
                model.addConstr(Y[i,p] - Y[i,q] - B[p,q,1,0] <= 0)
    for p in range(numMutations):
        for q in range(numMutations):
            model.addConstr(B[p,q,0,1] + B[p,q,1,0] + B[p,q,1,1] <= 2)


    objective = 0
    for j in range(numMutations):
        numZeros = 0
        numOnes  = 0
        for i in range(numCells):
            if I[i][j] == 0:
                numZeros += 1
                objective += logb1ma * Y[i,j]
            elif I[i][j] == 1:
                numOnes += 1
                objective += log1mba * Y[i,j]
            
        objective += numZeros * log1ma
        objective += numOnes * loga
        # objective -= 0 * (numZeros * log1m + numOnes * (np.log(alpha) + np.log((1-beta)/alpha)))

    model.setObjective(objective, GRB.MAXIMIZE)
    model.setParam('TimeLimit', time_limit)
    a = time.time()
    model.optimize()
    b = time.time()
    if model.status == GRB.Status.INFEASIBLE:
        print('The model is infeasible.')
        exit(0)

    # for i in range(numCells):
    #     sol_Y.append([nearestInt(float(Y[i,j].X)) for j in range(numMutations)])

    ###############
    sol_Y = np.zeros((numCells, numMutations))
    for i in range(numCells):
        for j in range(numMutations):
            # print(type(Y[i, j].X))
            # exit(0)
            sol_Y[i, j] =  Y[i, j].X > 0.5
            # sol_Y[i, j] =  Y[i, j].X * 100
    ###############

    status = {
        GRB.Status.OPTIMAL:'optimality',
        GRB.Status.TIME_LIMIT:'time_limit',
    }
    return np.array(sol_Y, dtype = np.int8), count_flips(I, I.shape[1] * [0], sol_Y), status[model.status], b-a


def PhISCS_B_external(matrix, beta=None, alpha=None, csp_solver_path=openwbo_path, time_limit = 3600):
    n, m = matrix.shape
    # par_fnRate = beta
    # par_fpRate = alpha
    par_fnWeight = 1
    par_fpWeight = 10

    Y = np.empty((n, m), dtype=np.int64)
    numVarY = 0
    map_y2ij = {}
    for i in range(n):
        for j in range(m):
            numVarY += 1
            map_y2ij[numVarY] = (i, j)
            Y[i, j] = numVarY

    X = np.empty((n, m), dtype=np.int64)
    numVarX = 0
    for i in range(n):
        for j in range(m):
            numVarX += 1
            X[i, j] = numVarY + numVarX

    B = np.empty((m, m, 2, 2), dtype=np.int64)
    numVarB = 0
    for p in range(m):
        for q in range(m):
            for i in range(2):
                for j in range(2):
                    numVarB += 1
                    B[p, q, i, j] = numVarY + numVarX + numVarB

    clauseHard = []
    clauseSoft = []
    numZero = 0
    numOne = 0
    numTwo = 0
    for i in range(n):
        for j in range(m):
            if matrix[i, j] == 0:
                numZero += 1
                cnf = "{} {}".format(par_fnWeight, -X[i, j])
                clauseSoft.append(cnf)
                cnf = "{} {}".format(-X[i, j], Y[i, j])
                clauseHard.append(cnf)
                cnf = "{} {}".format(X[i, j], -Y[i, j])
                clauseHard.append(cnf)
            elif matrix[i, j] == 1:
                numOne += 1
                cnf = "{} {}".format(par_fpWeight, -X[i, j])
                clauseSoft.append(cnf)
                cnf = "{} {}".format(X[i, j], Y[i, j])
                clauseHard.append(cnf)
                cnf = "{} {}".format(-X[i, j], -Y[i, j])
                clauseHard.append(cnf)
            elif matrix[i, j] == 2:
                numTwo += 1
                cnf = "{} {}".format(-1 * X[i, j], Y[i, j])
                clauseHard.append(cnf)
                cnf = "{} {}".format(X[i, j], -1 * Y[i, j])
                clauseHard.append(cnf)

    for i in range(n):
        for p in range(m):
            for q in range(p, m):
                # ~Yip v ~Yiq v Bpq11
                cnf = "{} {} {}".format(-Y[i, p], -Y[i, q], B[p, q, 1, 1])
                clauseHard.append(cnf)
                # Yip v ~Yiq v Bpq01
                cnf = "{} {} {}".format(Y[i, p], -Y[i, q], B[p, q, 0, 1])
                clauseHard.append(cnf)
                # ~Yip v Yiq v Bpq10
                cnf = "{} {} {}".format(-Y[i, p], Y[i, q], B[p, q, 1, 0])
                clauseHard.append(cnf)
                # ~Bpq01 v ~Bpq10 v ~Bpq11
                cnf = "{} {} {}".format(-B[p, q, 0, 1], -B[p, q, 1, 0], -B[p, q, 1, 1])
                clauseHard.append(cnf)

    hardWeight = numZero * par_fnWeight + numOne * par_fpWeight + 1

    outfile = "cnf.tmp"
    with open(outfile, "w") as out:
        out.write(
            "p wcnf {} {} {}\n".format(numVarY + numVarX + numVarB, len(clauseSoft) + len(clauseHard), hardWeight)
        )
        for cnf in clauseSoft:
            out.write("{} 0\n".format(cnf))
        for cnf in clauseHard:
            out.write("{} {} 0\n".format(hardWeight, cnf))

    finished_correctly = True
    a = time.time()
    command = "{} {}".format(csp_solver_path, outfile)
    with subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        try:
            output, error = proc.communicate(timeout = time_limit)
        except subprocess.TimeoutExpired:
            finished_correctly = False
            output_matrix = matrix
            flip_counts = np.zeros(4)
            termination_condition = 'time_limit'
    b = time.time()
    internal_time = b - a
    if finished_correctly:
        variables = output.decode().split("\n")[-2][2:].split(" ")
        O = np.empty((n, m), dtype=np.int8)
        numVar = 0
        for i in range(n):
            for j in range(m):
                if matrix[i, j] == 0:
                    if "-" in variables[numVar]:
                        O[i, j] = 0
                    else:
                        O[i, j] = 1
                elif matrix[i, j] == 1:
                    if "-" in variables[numVar]:
                        O[i, j] = 0
                    else:
                        O[i, j] = 1
                elif matrix[i, j] == 2:
                    if "-" in variables[numVar]:
                        O[i, j] = 0
                    else:
                        O[i, j] = 1
                numVar += 1

        output_matrix = O
        flip_counts = count_flips(matrix, matrix.shape[1] * [0], O)
        termination_condition = 'optimality'
    return output_matrix, flip_counts, termination_condition, internal_time


def top10_bad_entries_in_violations(D):
    def calc_how_many_violations_are_in(D, i, j):
        total = 0
        for p in range(D.shape[1]):
            if p == j:
                continue
            oneone = 0
            zeroone = 0
            onezero = 0
            founded = False
            for r in range(D.shape[0]):
                if D[r, p] == 1 and D[r, j] == 1:
                    oneone += 1
                    if r == i:
                        founded = True
                if D[r, p] == 0 and D[r, j] == 1:
                    zeroone += 1
                    if r == i:
                        founded = True
                if D[r, p] == 1 and D[r, j] == 0:
                    onezero += 1
                    if r == i:
                        founded = True
            if founded:
                total += oneone * zeroone * onezero
        return total

    violations = {}
    for r in range(D.shape[0]):
        for p in range(D.shape[1]):
            if D[r, p] == 0:
                violations[(r, p)] = calc_how_many_violations_are_in(D, r, p)

    for x in sorted(violations.items(), key=operator.itemgetter(1), reverse=True)[:10]:
        print(x[0], "(entry={}): how many gametes".format(D[x[0]]), x[1])

def PhISCS_B_timed(matrix, beta=None, alpha=None, time_limit = 3600):
    def returned_PhISCS_B(matrix, returned_dict):
        returned_dict["returned_value"] = PhISCS_B(matrix)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict["returned_value"] = "not filled yet"
    # Start foo as a process
    # multiprocessing.Pool(processes=1)
    p = multiprocessing.Process(target=returned_PhISCS_B, name="returned_PhISCS_B", args=(matrix, return_dict))
    p.start()
    p.join(time_limit)
    # If thread is active
    if p.is_alive():
        p.terminate()
        p.join()
        output_matrix = matrix
        flip_counts = np.zeros(4)
        termination_condition = 'time_limit'
        internal_time = 2 * time_limit
    else:
        output_matrix = return_dict["returned_value"][0]
        flip_counts = return_dict["returned_value"][1]
        termination_condition = 'optimality' #'time_limit'
        internal_time = return_dict["returned_value"][2]
    return output_matrix, flip_counts, termination_condition, internal_time


def PhISCS_B(matrix, beta=None, alpha=None):
    rc2 = RC2(WCNF())
    n, m = matrix.shape
    par_fnWeight = 1
    par_fpWeight = 10

    Y = np.empty((n, m), dtype=np.int64)
    numVarY = 0
    map_y2ij = {}
    for i in range(n):
        for j in range(m):
            numVarY += 1
            map_y2ij[numVarY] = (i, j)
            Y[i, j] = numVarY

    X = np.empty((n, m), dtype=np.int64)
    numVarX = 0
    for i in range(n):
        for j in range(m):
            numVarX += 1
            X[i, j] = numVarY + numVarX

    B = np.empty((m, m, 2, 2), dtype=np.int64)
    numVarB = 0
    for p in range(m):
        for q in range(m):
            for i in range(2):
                for j in range(2):
                    numVarB += 1
                    B[p, q, i, j] = numVarY + numVarX + numVarB

    for i in range(n):
        for j in range(m):
            if matrix[i, j] == 0:
                rc2.add_clause([-X[i, j]], weight=par_fnWeight)
                rc2.add_clause([-X[i, j], Y[i, j]])
                rc2.add_clause([X[i, j], -Y[i, j]])
            elif matrix[i, j] == 1:
                rc2.add_clause([-X[i, j]], weight=par_fpWeight)
                rc2.add_clause([X[i, j], Y[i, j]])
                rc2.add_clause([-X[i, j], -Y[i, j]])
            elif matrix[i, j] == 2:
                rc2.add_clause([-X[i, j], Y[i, j]])
                rc2.add_clause([X[i, j], -Y[i, j]])

    for i in range(n):
        for p in range(m):
            for q in range(p, m):
                rc2.add_clause([-Y[i, p], -Y[i, q], B[p, q, 1, 1]])
                rc2.add_clause([Y[i, p], -Y[i, q], B[p, q, 0, 1]])
                rc2.add_clause([-Y[i, p], Y[i, q], B[p, q, 1, 0]])
                rc2.add_clause([-B[p, q, 0, 1], -B[p, q, 1, 0], -B[p, q, 1, 1]])

    a = time.time()
    variables = rc2.compute()
    b = time.time()

    O = np.empty((n, m), dtype=np.int8)
    numVar = 0
    for i in range(n):
        for j in range(m):
            if matrix[i, j] == 0:
                if variables[numVar] < 0:
                    O[i, j] = 0
                else:
                    O[i, j] = 1
            elif matrix[i, j] == 1:
                if variables[numVar] < 0:
                    O[i, j] = 0
                else:
                    O[i, j] = 1
            elif matrix[i, j] == 2:
                if variables[numVar] < 0:
                    O[i, j] = 0
                else:
                    O[i, j] = 1
            numVar += 1
    
    return O, count_flips(matrix, matrix.shape[1] * [0], O), b - a


def rename(new_name):
    def decorator(f):
        f.__name__ = new_name
        return f

    return decorator


def get_k_partitioned_PhISCS(k):
    @rename(f"partitionedPhISCS_{k}")
    def partitioned_PhISCS(x):
        ans = 0
        for i in range(x.shape[1] // k):
            ans += myPhISCS_I(x[:, i * k : (i + 1) * k])
        if x.shape[1] % k >= 2:
            ans += myPhISCS_I(x[:, ((x.shape[1] // k) * k) :])
        return ans

    return partitioned_PhISCS


def from_interface_to_method(bounding_alg):
    def run_func(x):
        bounding_alg.reset(x)
        ret = bounding_alg.get_bound(sp.lil_matrix(x.shape, dtype=np.int8))
        return ret, bounding_alg.get_name()

    run_func.core = bounding_alg  # include arguments in the name
    run_func.__name__ = bounding_alg.get_name()
    return run_func


def upper_bound_2_sat_timed(matrix, time_limit):
    args = {"matrix" : matrix}
    result = timed_run(upper_bound_2_sat, args, time_limit=time_limit)
    if result["termination_condition"] == "success":
        output = result["output"]
        output = (output[0], output[1], "optimality", output[2])
    elif result["termination_condition"] == "time_limit":
        output = (None, (0,0,0,0), "time_limit", time_limit)
    return output

def zero_or_na(vec, na_value = 2): # todo: a more pragmatic way to set na_value
    return np.logical_or(vec == 0, vec == na_value)

def get_effective_matrix(I, delta01, delta21, change20 = False):
    x = np.array(I + delta01, dtype = np.int8)
    if delta21 is not None:
        na_indices = delta21.nonzero()
        x[na_indices] = 1# should have been (but does not accept): x[na_indices] = delta21[na_indices]
    if change20:
        x[ x==2 ] = 0
    return x

def make_2sat_model(matrix, threshold = 0, coloring = None, na_value = 2, eps = None ):
    if eps is None:
        eps = 1 / (matrix.shape[0] + matrix.shape[1])
    hard_cnst_num = 0
    soft_cnst_num = 0
    rc2 = RC2(WCNF())
    n, m = matrix.shape

    #variables for each zero
    F = - np.ones((n, m), dtype=np.int64)
    num_var_F = 0
    map_f2ij = {}

    # variables for pair of columns
    num_var_B = 0
    map_b2pq = {}
    B = - np.ones((m, m), dtype=np.int64)
    #
    # for i in range(n):
    #     for j in range(m):
    #         if matrix[i, j] == 0:
    #             num_var_F += 1
    #             map_f2ij[num_var_F] = (i, j)
    #             F[i, j] = num_var_F
    #             rc2.add_clause([-F[i,j]], weight = 1)
    #             soft_cnst_num += 1
    # if version == 1:
    #

    col_pair = None
    pair_cost = 0
    for p in range(m):
        for q in range(p+1, m):
            if np.any(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 1)): # p and q has intersection
                r01 = np.nonzero(np.logical_and(zero_or_na(matrix[:, p]), matrix[:, q] == 1))[0]
                r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, zero_or_na(matrix[:, q])))[0]
                cost = min(len(r01), len(r10))
                if cost > pair_cost: # keep best pair to return as auxiliary info
                    col_pair = (p, q)
                    pair_cost = cost

                if coloring is not None and coloring[p] != coloring[q]:
                    continue
                direct_formulation_cost = len(r01) * len(r10)
                indirect_formulation_cost = len(r01) + len(r10)
                if  direct_formulation_cost * threshold <= indirect_formulation_cost: # use direct formulation
                    for a, b in itertools.product(r01, r10):
                        for x, y in [(a, p), (b, q)]: # make sure the variables for this are made
                            if F[x, y] < 0:
                                num_var_F += 1
                                map_f2ij[num_var_F + num_var_B] = (x, y)
                                F[x, y] = num_var_F + num_var_B
                                if matrix[x, y] == 0: # do not add weight for na s
                                    w = 1
                                else:
                                    w = eps
                                if w > 0:
                                    rc2.add_clause([-F[x,y]], weight = w)
                                    soft_cnst_num += 1
                        rc2.add_clause([F[a, p], F[b, q]]) # at least one of them should be flipped
                        hard_cnst_num += 1
                else:# use indirect formulation
                    if cost > 0 :
                        num_var_B += 1
                        map_b2pq[num_var_F + num_var_B] = (p, q)
                        B[p, q] = num_var_B + num_var_F
                        queue = itertools.chain(
                            itertools.product(r01, [1]),
                            itertools.product(r10, [-1]),
                        )
                        for row, sign in queue:
                            col = p if sign == 1 else q
                            for x, y in [(row, col)]:  # make sure the variables for this are made
                                if F[x, y] < 0:
                                    num_var_F += 1
                                    map_f2ij[num_var_F + num_var_B] = (x, y)
                                    F[x, y] = num_var_F + num_var_B
                                    if matrix[x, y] == 0:  # do not add weight for na s
                                        w = 1
                                    else:
                                        w = eps
                                    if w > 0:
                                        rc2.add_clause([-F[x, y]], weight=w)
                                        soft_cnst_num += 1
                            rc2.add_clause([F[row, col], sign * B[p, q]])
                            hard_cnst_num += 1
                            # either the all the zeros in col1 should be fliped
                            # OR this pair is already covered
                # else:
                #     raise Exception("version not implemented")
                # exit(0)
    # print(num_var_F, num_var_B, soft_cnst_num, hard_cnst_num, threshold, sep="\t")
    # print(num_var_F, num_var_B, hard_cnst_num, sep = "\t", end="\t")

    return rc2, col_pair, map_f2ij, map_b2pq

def get_clustering(matrix, na_value = 2):
    from sklearn.cluster import MiniBatchKMeans, KMeans
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    X = matrix.T
    X[X == na_value] = 0.5
    # k_means = KMeans(init='k-means++', n_clusters=2, n_init =10)
    k_means = KMeans(init='random', n_clusters=2, n_init=2)
    # k_means = KMeans(init='random', n_clusters=2)
    k_means.fit(X)
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    coloring = np.array(k_means_labels)
    # print(repr(coloring))
    # coloring = np.ones(coloring.shape)
    # coloring = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    # coloring = np.random.randint(2, size=coloring.shape)
    return coloring

def upper_bound_2_sat(matrix, threshold, version):
    """
    This algorithm is based on iterative usage of weighted 2-sat solver.
    It runs in polynomial time (TODO complexity) and is theoretically guaranteed to give an upper bound
    :param threshold:
    :param matrix:
    :return:
    """
    # print("call ", version)
    if version == 0:
        coloring = None
    elif version == 1:
        coloring = get_clustering(matrix)
    else:
        raise NotImplementedError("version?")
    rc2, col_pair, map_f2ij, map_b2pq = make_2sat_model(matrix, threshold = threshold, coloring = coloring)
    icf = col_pair is None
    # print(col_pair, num_var_F)
    nf = 0
    if icf:
        return matrix.copy(), (0, 0, 0, 0), 0
    else:
        a = time.time()
        variables = rc2.compute()
        b = time.time()

        # print(variables)
        # exit(0)
        O = matrix.copy()
        O = O.astype(np.int8)
        for var_ind in range(len(variables)):
            if 0 < variables[var_ind] and variables[var_ind] in map_f2ij: # if 0 or 2 make it one
            # if 0 < variables[var_ind] :
                O[map_f2ij[variables[var_ind]]] = 1
                nf += 1
            if 0 > variables[var_ind] and matrix[map_f2ij[-variables[var_ind]]] == 2 : # if 2 make it one
                O[map_f2ij[-variables[var_ind]]] = 0
                nf += 1
                # print("flip ", map_f2ij[variables[var_ind]])
        cntflip = list(count_flips(matrix, matrix.shape[1] * [0], O)) # second argument means no columns has been removed


        # *** Even if the first version was one I am changing it so that solve the matrix
        Orec, cntfliprec, timerec = upper_bound_2_sat(O, threshold=threshold, version = 0)
        for ind in range(len(cntflip)):
            cntflip[ind] += cntfliprec[ind]
        return Orec, tuple(cntflip), timerec + b - a


def save_matrix_to_file(filename, numpy_input):
    n, m = numpy_input.shape
    dfout = pd.DataFrame(numpy_input)
    dfout.columns = ['mut'+str(j) for j in range(m)]
    dfout.index = ['cell'+str(j) for j in range(n)]
    dfout.index.name = 'cellIDxmutID'
    dfout.to_csv(filename, sep='\t')


def draw_tree_muts_in_edges(filename):
    bulkfile=''
    addBulk=False
    add_cells=False

    import numpy as np
    import pandas as pd
    import pygraphviz as pyg
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

    def contains(col1, col2):
        for i in range(len(col1)):
            if not col1[i] >= col2[i]:
                return False
        return True

    df = pd.read_csv(filename, sep='\t').set_index('cellID/mutID')
    splitter_mut = '\n'
    matrix = df.values
    names_mut = list(df.columns)

    i = 0
    while i < matrix.shape[1]:
        j = i + 1
        while j < matrix.shape[1]:
            if np.array_equal(matrix[:,i], matrix[:,j]):
                matrix = np.delete(matrix, j, 1)
                x = names_mut.pop(j)
                names_mut[i] += splitter_mut + x
                j -= 1
            j += 1
        i += 1

    rows = matrix.shape[0]
    cols = matrix.shape[1]
    dimensions = np.sum(matrix, axis=0)
    indices = np.argsort(dimensions)
    dimensions = np.sort(dimensions)
    names_mut = [names_mut[indices[i]] for i in range(cols)]

    G = nx.DiGraph()
    G.add_node(cols)
    G.add_node(cols-1)
    G.add_edge(cols, cols-1, label=names_mut[cols-1])
    node_mud = {}
    node_mud[names_mut[cols-1]] = cols-1

    i = cols - 2
    while i >= 0:
        if dimensions[i] == 0:
            break
        attached = False
        for j in range(i+1, cols):
            if contains(matrix[:, indices[j]], matrix[:, indices[i]]):
                G.add_node(i)
                G.add_edge(node_mud[names_mut[j]], i, label=names_mut[i])
                node_mud[names_mut[i]] = i
                attached = True
                break
        if not attached:
            G.add_node(i)
            G.add_edge(cols, i, label=names_mut[i])
            node_mud[names_mut[i]] = i
        i -=1

    clusters = {}
    for node in G:
        if node == cols:
            G._node[node]['label'] = '<<b>germ<br/>cells</b>>'
            G._node[node]['fontname'] = 'Helvetica'
            G._node[node]['width'] = 0.4
            G._node[node]['style'] = 'filled'
            G._node[node]['penwidth'] = 3
            G._node[node]['fillcolor'] = 'gray60'
            continue
        untilnow_mut = []
        sp = nx.shortest_path(G, cols, node)
        for i in range(len(sp)-1):
            untilnow_mut += G.get_edge_data(sp[i], sp[i+1])['label'].split(splitter_mut)
        untilnow_cell = df.loc[(df[untilnow_mut] == 1).all(axis=1) & \
                               (df[[x for x in df.columns if x not in untilnow_mut]] == 0).all(axis=1)].index
        if len(untilnow_cell) > 0:
            clusters[node] = '\n'.join(untilnow_cell)
        else:
            clusters[node] = '-'
        
        if add_cells:
            G._node[node]['label'] = clusters[node]
        else:
            G._node[node]['label'] = ''
            G._node[node]['shape'] = 'circle'
        G._node[node]['fontname'] = 'Helvetica'
        G._node[node]['width'] = 0.4
        G._node[node]['style'] = 'filled'
        G._node[node]['penwidth'] = 2
        G._node[node]['fillcolor'] = 'gray90'
    i = 1
    for k, v in clusters.items():
        if v == '-':
            clusters[k] = i*'-'
            i += 1

    header = ''
    if addBulk:
        vafs = {}
        bulkMutations = readMutationsFromBulkFile(bulkfile)
        sampleIDs = bulkMutations[0].getSampleIDs()
        for mut in bulkMutations:
            temp_vaf = []
            for sample in sampleIDs:
                temp_vaf.append(str(mut.getVAF(sampleID=sample)))
            vafs[mut.getID()] = '<font color="blue">'+','.join(temp_true)+'</font>'        
        for edge in G.edges():
            temp = []
            for mut in G.get_edge_data(edge[0],edge[1])['label'].split(splitter_mut):
                mut = '<u>' + mut + '</u>' + ': ' + vafs_true[mut] + '; ' + vafs_noisy[mut]
                temp.append(mut)
            temp = '<' + '<br/>'.join(temp) + '>'
            G.get_edge_data(edge[0],edge[1])['label'] = temp

        for mut in bulkMutations:
            try:
                isatype = mut.getINFOEntryStringValue('ISAVtype')
                header += mut.getID() + ': ' + isatype + '<br/>'
            except:
                pass
    
    temp = df.columns[(df==0).all(axis=0)]
    if len(temp) > 0:
        header += 'Became Germline: ' + ','.join(temp) + '<br/>'
    
    '''
    with open(filename[:-len('.CFMatrix')]+'.log') as fin:
        i = 0
        for line in fin:
            i += 1
            if i > 10 and i < 18:
                header += line.rstrip() + '<br/>'
    '''

    H = nx.relabel_nodes(G, clusters)
    html = '''<{}>'''.format(header)
    H.graph['graph'] = {'label':html, 'labelloc':'t', 'resolution':300, 'fontname':'Helvetica', 'fontsize':8}
    H.graph['node'] = {'fontname':'Helvetica', 'fontsize':8}
    H.graph['edge'] = {'fontname':'Helvetica', 'fontsize':8}
    
    mygraph = to_agraph(H)
    mygraph.layout(prog='dot')
    outputpath = filename[:-len('.CFMatrix')]
    mygraph.draw('{}.edges.png'.format(outputpath))


def draw_tree_muts_in_nodes(filename):
    addBulk=False
    bulkfile=''
    import numpy as np
    import pandas as pd
    import pygraphviz as pyg

    graph = pyg.AGraph(strict=False, directed=True)
    font_name = 'Avenir'

    class Node:
        def __init__(self, name, parent):
            self.name = name
            self.parent = parent
            self.children = []
            if parent:
                parent.children.append(self)

    def print_tree(node):
        graph.add_node(node.name, label=node.name, fontname=font_name, color='black', penwidth=3.5)
        for child in node.children:
            graph.add_edge(node.name, child.name)
            print_tree(child)

    def contains(col1, col2):
        for i in range(len(col1)):
            if not col1[i] >= col2[i]:
                return False
        return True

    def write_tree(matrix, names):
        i = 0
        while i < matrix.shape[1]:
            j = i + 1
            while j < matrix.shape[1]:
                if np.array_equal(matrix[:,i], matrix[:,j]):
                    matrix = np.delete(matrix, j, 1)
                    x = names.pop(j)
                    names[i] += '<br/><br/>' + x
                    j -= 1
                j += 1
            names[i] = '<'+names[i]+'>'
            i += 1

        rows = len(matrix)
        cols = len(matrix[0])
        dimensions = np.sum(matrix, axis=0)
        # ordered indeces
        indeces = np.argsort(dimensions)
        dimensions = np.sort(dimensions)
        mutations_name = []
        for i in range(cols):
            mutations_name.append(names[indeces[i]])

        root = Node(mutations_name[-1], None)
        mut_nod = {}
        mut_nod[mutations_name[cols-1]] = root

        i = cols - 2
        while i >=0:
            if dimensions[i] == 0:
                break
            attached = False
            for j in range(i+1, cols):
                if contains(matrix[:, indeces[j]], matrix[:, indeces[i]]):
                    node = Node(mutations_name[i], mut_nod[mutations_name[j]])
                    mut_nod[mutations_name[i]] = node
                    attached = True
                    break
            if not attached:
                node = Node(mutations_name[i], root)
                mut_nod[mutations_name[i]] = node
            i -=1
        print_tree(root)

    if addBulk:
        vafs = {}
        bulkMutations = readMutationsFromBulkFile(bulkfile)
        sampleIDs = bulkMutations[0].getSampleIDs()
        for mut in bulkMutations:
            temp_vaf = []
            for sample in sampleIDs:
                temp_vaf.append('<font color="blue">' + str(mut.getVAF(sampleID=sample)) + '</font>')
            vafs[mut.getID()] = '{} ({})'.format(mut.getID(), ','.join(temp_vaf))

    inp = np.genfromtxt(filename, skip_header=1, delimiter='\t')
    with open(filename, 'r') as fin:
        if addBulk:
            mutation_names = [vafs[x] for x in fin.readline().strip().split('\t')[1:]]
        else:
            mutation_names = fin.readline().strip().split('\t')[1:]
    sol_matrix = np.delete(inp, 0, 1)
    write_tree(sol_matrix, mutation_names)
    graph.layout(prog='dot')
    outputpath = filename[:-len('.CFMatrix')]
    graph.draw('{}.nodes.png'.format(outputpath))


if __name__ == '__main__':
    hash = 1151136
    a = read_matrix_from_file(f"../../solution_{hash}_BnB_1_two_sat_-1_0_0.CFMatrix")



    b = read_matrix_from_file(f"../../solution_{hash}_PhISCS_B_timed_.CFMatrix")
    c = read_matrix_from_file(f"../../solution_{hash}_PhISCS_I_.CFMatrix")
    print(a.shape)
    # a = a[:, :]
    # b = b[:, :]
    # print(np.nonzero(np.sum(a!=b, axis = 0)))
    print(np.sum(a!=b))
    print(np.sum(b!=c))
    print(np.sum(a!=c))
    exit(0)

    # n = 20
    # m = 5
    # x = np.random.randint(3, size=(n, m))
    # x = np.array([
    #     [0, 0, 1, 0, 1],
    #     [1, 0, 1, 1, 1],
    #     [1, 0, 1, 1, 1],
    #     [0, 1, 0, 1, 0],
    #     [0, 0, 1, 1, 1],
    #     [0, 1, 1, 0, 0]
    # ])

    x = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")

    x = x[:50, :50]
    for i in range(x.shape[0]):
        x[i, np.random.randint(x.shape[1])] = 2
    # print(repr(x))

    result = PhISCS_B(x)
    print(result[1:], end="\n\n")
    #
    # result = PhISCS_I(x, beta = 0.99)
    # print(result[1:], end="\n\n")
    #
    # result = PhISCS_I(x, beta=0.90)
    # print(result[1:], end="\n\n")
    #
    # result = PhISCS_I(x, beta = 0.99)
    # print(result[1:], end="\n\n")

    result = upper_bound_2_sat(x, threshold=0, version=0)
    print(result[1:], end="\n\n")


    # result = upper_bound_2_sat(x, threshold=1)
    # print(result, end="\n\n")
    #
    # # result = PhISCS_B_2_sat_timed(x, time_limit=2)
    # # print(result)
    # #
    # # result = PhISCS_B_timed(x, time_limit=2)
    # # print(result)

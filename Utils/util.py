from Utils.const import *



def print_line(depth=1, shift=1):
    """A debugging tool!  """
    for i in range(shift, depth + shift):
        info = inspect.stack()[i]
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


# def myPhISCS_B(x):
#     solution, (f_0_1_b, f_1_0_b, f_2_0_b, f_2_1_b), cb_time = PhISCS_B(x, beta=0.90, alpha=0.00001)
#     nf = len(np.where(solution != x)[0])
#     return nf


def myPhISCS_I(x):
    ret = PhISCS_I(x, beta=0.90, alpha=0.00001)
    solution = ret[0]
    nf = len(np.where(solution != x)[0])
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


def PhISCS_I(I, beta=0.97, alpha=0.00001, time_limit = 3600):
    def nearestInt(x):
        return int(x+0.5)

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
                objective += np.log(beta/(1-alpha)) * Y[i,j]
            elif I[i][j] == 1:
                numOnes += 1
                objective += np.log((1-beta)/alpha) * Y[i,j]
            
        objective += numZeros * np.log(1-alpha)
        objective += numOnes * np.log(alpha)
        objective -= 0 * (numZeros * np.log(1-alpha) + numOnes * (np.log(alpha) + np.log((1-beta)/alpha)))

    model.setObjective(objective, GRB.MAXIMIZE)
    model.setParam('TimeLimit', time_limit)
    a = time.time()
    model.optimize()
    b = time.time()
    if model.status == GRB.Status.INFEASIBLE:
        print('The model is infeasible.')
        exit(0)

    for i in range(numCells):
        sol_Y.append([nearestInt(float(Y[i,j].X)) for j in range(numMutations)])

    status = {
        GRB.Status.OPTIMAL:'optimality',
        GRB.Status.TIME_LIMIT:'time_limit',
    }
    return np.array(sol_Y), count_flips(I, I.shape[1] * [0], sol_Y), status[model.status], b-a


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


def PhISCS_B_2_sat_timed(matrix, time_limit):
    args = {"matrix" : matrix}
    result = timed_run(PhISCS_B_2_sat, args, time_limit=time_limit)
    if result["termination_condition"] == "success":
        output = result["output"]
        output = (output[0], output[1], "optimality", output[2])
    elif result["termination_condition"] == "time_limit":
        output = (None, (0,0,0,0), "time_limit", time_limit)
    return output

def PhISCS_B_2_sat(matrix,):
    """
    This algorithm is based on iterative usage of weighted 2-sat solver.
    It runs in polynomial time (TODO complexity) and is theoretically guaranteed to give an upper bound
    :param matrix:
    :return:
    """
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

    icf = True
    for p in range(m):
        for q in range(m):
            if p != q and np.any(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 1)):
                r01 = np.nonzero(np.logical_and(matrix[:, p] == 0, matrix[:, q] == 1))[0]
                r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 0))[0]
                if len(r01) * len(r10) > 0:
                    icf = False
                for a, b in itertools.product(r01, r10):
                    rc2.add_clause([F[a, p], F[b, q]]) # at least one of them should be flipped
    # print(icf)
    if icf:
        return matrix.copy(), (0, 0, 0, 0), 0
    else:
        a = time.time()
        variables = rc2.compute()
        b = time.time()


        O = matrix.copy()
        O = O.astype(np.int8)
        for var_ind in range(len(variables)):
            if variables[var_ind] > 0:
                O[map_f2ij[variables[var_ind]]] = 1

    Orec, cntfliprec, timerec = PhISCS_B_2_sat(O)
    cntflip = list(count_flips(matrix, matrix.shape[1] * [0], O))
    for ind in range(len(cntflip)):
        cntflip[ind] += cntfliprec[ind]
    return Orec, tuple(cntflip), timerec + b - a



if __name__ == '__main__':
    # n = 6
    # m = 5
    # x = np.random.randint(2, size=(n, m))
    x = np.array([[0, 0, 1, 0, 1],
       [1, 0, 1, 1, 1],
       [1, 0, 1, 1, 1],
       [0, 1, 0, 1, 0],
       [0, 0, 1, 1, 1],
       [0, 1, 1, 0, 0]])

    print(repr(x))
    result = PhISCS_B_2_sat(x)
    print(result)

    # result = PhISCS_B_2_sat_timed(x, time_limit=2)
    # print(result)

    result = PhISCS_B_timed(x, time_limit=2)
    print(result)

from utils.const import *
from utils.util import *



def PhISCS_B_nf(x):
    solution, cb_time = PhISCS_B(x, beta=0.97, alpha=0.00001)
    return count_nf(solution, x)


def PhISCS_I_nf(x):
    ret = PhISCS_I(x, beta=0.99, alpha=0.00001)
    solution = ret[0]
    return count_nf(solution, x)


def PhISCS_I(I, beta=0.99, alpha=0.00001, time_limit=None):

    logb1ma, log1mba, log1ma, loga = \
        np.log(beta / (1 - alpha)), np.log((1 - beta) / alpha), np.log((1 - alpha)), np.log(alpha)

    if - log1mba / logb1ma > 70:
        logb1ma = -1
        log1mba = 70

    num_cells, num_mutations = I.shape

    model = Model('PhISCS_ILP')
    model.Params.LogFile = ''
    model.Params.OutputFlag = 0
    model.Params.Threads = 1


    Y = {}
    for c in range(num_cells):
        for m in range(num_mutations):
            Y[c, m] = model.addVar(vtype=GRB.BINARY, name='Y({0},{1})'.format(c, m))
    B = {}
    for p in range(num_mutations):
        for q in range(num_mutations):
            B[p, q, 1, 1] = model.addVar(vtype=GRB.BINARY, obj=0, name='B[{0},{1},1,1]'.format(p, q))
            B[p, q, 1, 0] = model.addVar(vtype=GRB.BINARY, obj=0, name='B[{0},{1},1,0]'.format(p, q))
            B[p, q, 0, 1] = model.addVar(vtype=GRB.BINARY, obj=0, name='B[{0},{1},0,1]'.format(p, q))
    model.update()
    for i in range(num_cells):
        for p in range(num_mutations):
            for q in range(num_mutations):
                model.addConstr(Y[i, p] + Y[i, q] - B[p, q, 1, 1] <= 1)
                model.addConstr(-Y[i, p] + Y[i, q] - B[p, q, 0, 1] <= 0)
                model.addConstr(Y[i, p] - Y[i, q] - B[p, q, 1, 0] <= 0)
    for p in range(num_mutations):
        for q in range(num_mutations):
            model.addConstr(B[p, q, 0, 1] + B[p, q, 1, 0] + B[p, q, 1, 1] <= 2)

    objective = 0
    for j in range(num_mutations):
        numZeros = 0
        numOnes = 0
        for i in range(num_cells):
            if I[i][j] == 0:
                numZeros += 1
                objective += logb1ma * Y[i, j]
            elif I[i][j] == 1:
                numOnes += 1
                objective += log1mba * Y[i, j]

        objective += numZeros * log1ma
        objective += numOnes * loga
        # objective -= 0 * (numZeros * log1m + numOnes * (np.log(alpha) + np.log((1-beta)/alpha)))

    model.setObjective(objective, GRB.MAXIMIZE)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    a = time.time()
    model.optimize()
    b = time.time()
    if model.status == GRB.Status.INFEASIBLE:
        print('The model is infeasible.')
        exit(0)

    sol_Y = np.zeros((num_cells, num_mutations))
    for i in range(num_cells):
        for j in range(num_mutations):
            sol_Y[i, j] = Y[i, j].X > 0.5

    status = {
        GRB.Status.OPTIMAL: 'optimality',
        GRB.Status.TIME_LIMIT: 'time_limit',
    }
    return np.array(sol_Y, dtype=np.int8), status[model.status], b - a


def PhISCS_B_external(matrix, beta=None, alpha=None, csp_solver_path=openwbo_path, time_limit=3600):
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
            output, error = proc.communicate(timeout=time_limit)
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
        termination_condition = 'optimality'
    return output_matrix, termination_condition, internal_time


def PhISCS_B(matrix, beta=None, alpha=None, na_value=2):
    rc2 = RC2(WCNF())
    n, m = matrix.shape
    par_fnWeight = 1
    par_fpWeight = int(beta/alpha)

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
            elif matrix[i, j] == na_value:
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
            elif matrix[i, j] == na_value:
                if variables[numVar] < 0:
                    O[i, j] = 0
                else:
                    O[i, j] = 1
            numVar += 1

    return O, b - a



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

if __name__ == '__main__':
    n, m = 6, 6
    x = np.random.randint(2, size=(n, m))
    output = timed_run(PhISCS_B, {"matrix": x, "alpha":0.0001, "beta":0.9}, 5)
    print(output)
    # (sol, internal_time = output["output"]
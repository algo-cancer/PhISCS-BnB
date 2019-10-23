import os
import subprocess
import numpy as np
import random, math
import time
from gurobipy import *
from const import *

def is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I):
    def sort_bin(a):
        b = np.transpose(a)
        b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
        idx = np.argsort(b_view.ravel())[::-1]
        c = b[idx]
        return np.transpose(c), idx

    O, idx = sort_bin(I)
    #TODO: delete duplicate columns
    #print(O, '\n')
    Lij = np.zeros(O.shape, dtype=int)
    for i in range(O.shape[0]):
        maxK = 0
        for j in range(O.shape[1]):
            if O[i,j] == 1:
                Lij[i,j] = maxK
                maxK = j+1
    #print(Lij, '\n')
    Lj = np.amax(Lij, axis=0)
    #print(Lj, '\n')
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            if O[i,j] == 1:
                if Lij[i,j] != Lj[j]:
                    return False, (idx[j], idx[Lj[j]-1])
    return True, (None,None)


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


def get_data_by_ms(n, m, seed, fn, fp, na, ms_path):
    def make_noisy_by_fn(data, fn, fp, na):
        n, m = data.shape
        data2 = -1*np.ones(shape=(n, m)).astype(int)
        countFP = 0
        countFN = 0
        countNA = 0
        countOneZero = 0
        indexNA = []
        changedBefore = []
        for i in range(n):
            for j in range(m):
                indexNA.append([i,j])
                countOneZero = countOneZero + 1
        random.shuffle(indexNA)
        nas = math.ceil(countOneZero * na)
        for i in range(int(nas)):
            [a,b] = indexNA[i]
            changedBefore.append([a,b])
            data2[a][b] = 2
            countNA = countNA+1
        for i in range(n):
            for j in range(m):
                if data2[i][j] != 2:
                    if data[i][j] == 1:
                        if toss(fn):
                            data2[i][j] = 0
                            countFN = countFN+1
                        else:
                            data2[i][j] = data[i][j]
                    elif data[i][j] == 0:
                        if toss(fp):
                            data2[i][j] = 1
                            countFP = countFP+1
                        else:
                            data2[i][j] = data[i][j]
        return data2, (countFN,countFP,countNA)

    def toss(p):
        return True if np.random.random() < p else False

    def build_ground_by_ms(n, m, seed):
        command = '{ms} {n} 1 -s {m} -seeds 7369 217 {r} | tail -n {n}'.format(ms=ms_path, n=n, m=m, r=seed)
        result = os.popen(command).read()
        data = np.empty((n,m), dtype=int)
        i = 0
        for s in result.split('\n'):
            j = 0
            for c in list(s):
                data[i,j] = int(c)
                j += 1
            i += 1
        return data
    
    ground = build_ground_by_ms(n, m, seed)
    if is_conflict_free_farid(ground):
        noisy, (countFN,countFP,countNA) = make_noisy_by_fn(ground, fn, fp, na)
        if not is_conflict_free_farid(noisy):
            return ground, noisy, (countFN,countFP,countNA)
    else:
        return get_data_by_ms(n, m, seed+1, fn, fp, na, ms_path)

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


def PhISCS_I(I, beta, alpha):
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    def nearestInt(x):
        return int(x+0.5)

    maxMutationsToEliminate = 0
    numCells, numMutations = I.shape
    sol_Y = []
    sol_K = []
    with HiddenPrints():
        model = Model('PhISCS_ILP')
        model.Params.LogFile = ''
        model.Params.Threads = 1
        # model.setParam('TimeLimit', 10*60)
        
        Y = {}
        for c in range(numCells):
            for m in range(numMutations):
                    Y[c, m] = model.addVar(vtype=GRB.BINARY, name='Y({0},{1})'.format(c, m))
        B = {}
        for p in range(numMutations+1):
            for q in range(numMutations+1):
                B[p, q, 1, 1] = model.addVar(vtype=GRB.BINARY, obj=0,
                                                name='B[{0},{1},1,1]'.format(p, q))
                B[p, q, 1, 0] = model.addVar(vtype=GRB.BINARY, obj=0,
                                                name='B[{0},{1},1,0]'.format(p, q))
                B[p, q, 0, 1] = model.addVar(vtype=GRB.BINARY, obj=0,
                                                name='B[{0},{1},0,1]'.format(p, q))
        K = {}
        for m in range(numMutations+1):
            K[m] = model.addVar(vtype=GRB.BINARY, name='K[{0}]'.format(m))
        model.addConstr(K[numMutations] == 0)
        model.update()

        model.addConstr(quicksum(K[m] for m in range(numMutations)) <= maxMutationsToEliminate)
        for i in range(numCells):
            for p in range(numMutations):
                for q in range(numMutations):
                    model.addConstr(Y[i,p] + Y[i,q] - B[p,q,1,1] <= 1)
                    model.addConstr(-Y[i,p] + Y[i,q] - B[p,q,0,1] <= 0)
                    model.addConstr(Y[i,p] - Y[i,q] - B[p,q,1,0] <= 0)
        for p in range(numMutations+1):
            model.addConstr(B[p,numMutations, 1, 0] == 0)
        for p in range(numMutations):
            for q in range(numMutations):
                model.addConstr(B[p,q,0,1] + B[p,q,1,0] + B[p,q,1,1] <= 2 + K[p] + K[q])

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
            objective -= K[j] * (numZeros * np.log(1-alpha) + numOnes * (np.log(alpha) + np.log((1-beta)/alpha)))

        model.setObjective(objective, GRB.MAXIMIZE)
        a = time.time()
        model.optimize()
        b = time.time()

        if model.status == GRB.Status.INFEASIBLE:
            print('The odel is infeasible.')
            exit(0)

        removedMutsIDs = []
        for j in range(numMutations):
            sol_K.append(nearestInt(float(K[j].X)))
            if sol_K[j] == 1:
                removedMutsIDs.append(mutIDs[j])
        
        for i in range(numCells):
            sol_Y.append([nearestInt(float(Y[i,j].X)) for j in range(numMutations)])

    return np.array(sol_Y), count_flips(I, sol_K, sol_Y), b-a


def PhISCS_B(matrix, beta, alpha, csp_solver_path):
    n,m = matrix.shape
    par_fnRate = beta
    par_fpRate = alpha
    par_fnWeight = 1
    par_fpWeight = 10

    Y = np.empty((n,m), dtype=np.int64)
    numVarY = 0
    map_y2ij = {}
    for i in range(n):
        for j in range(m):
            numVarY += 1
            map_y2ij[numVarY] = (i, j)
            Y[i,j] = numVarY

    X = np.empty((n,m), dtype=np.int64)
    numVarX = 0
    for i in range(n):
        for j in range(m):
            numVarX += 1
            X[i,j] = numVarY + numVarX

    B = np.empty((n,m,2,2), dtype=np.int64)
    numVarB = 0
    for p in range(m):
        for q in range(m):
            for i in range(2):
                for j in range(2):
                    numVarB += 1
                    B[p,q,i,j] = numVarY + numVarX + numVarB;
    
    clauseHard = []
    clauseSoft = []
    numZero = 0
    numOne = 0
    numTwo = 0
    for i in range(n):
        for j in range(m):
            if matrix[i,j] == 0:
                numZero += 1
                cnf = '{} {}'.format(par_fnWeight, -X[i,j])
                clauseSoft.append(cnf)
                cnf = '{} {}'.format(-X[i,j], Y[i,j])
                clauseHard.append(cnf)
                cnf = '{} {}'.format(X[i,j], -Y[i,j])
                clauseHard.append(cnf)
            elif matrix[i,j] == 1:
                numOne += 1
                cnf = '{} {}'.format(par_fpWeight, -X[i,j])
                clauseSoft.append(cnf)
                cnf = '{} {}'.format(X[i,j], Y[i,j])
                clauseHard.append(cnf)
                cnf = '{} {}'.format(-X[i,j], -Y[i,j])
                clauseHard.append(cnf)
            elif matrix[i,j] == 2:
                numTwo += 1
                cnf = '{} {}'.format(-1*X[i,j], Y[i,j])
                clauseHard.append(cnf)
                cnf = '{} {}'.format(X[i,j], -1*Y[i,j])
                clauseHard.append(cnf)

    for i in range(n):
        for p in range(m):
            for q in range(p, m):
                #~Yip v ~Yiq v Bpq11
                cnf = '{} {} {}'.format(-Y[i,p], -Y[i,q], B[p,q,1,1])
                clauseHard.append(cnf)
                #Yip v ~Yiq v Bpq01
                cnf = '{} {} {}'.format(Y[i,p], -Y[i,q], B[p,q,0,1])
                clauseHard.append(cnf)
                #~Yip v Yiq v Bpq10
                cnf = '{} {} {}'.format(-Y[i,p], Y[i,q], B[p,q,1,0])
                clauseHard.append(cnf)
                #~Bpq01 v ~Bpq10 v ~Bpq11
                cnf = '{} {} {}'.format(-B[p,q,0,1], -B[p,q,1,0], -B[p,q,1,1])
                clauseHard.append(cnf)

    hardWeight = numZero * par_fnWeight + numOne * par_fpWeight + 1

    outfile = 'cnf.tmp'
    with open(outfile, 'w') as out:
        out.write('p wcnf {} {} {}\n'.format(numVarY+numVarX+numVarB, len(clauseSoft)+len(clauseHard), hardWeight))
        for cnf in clauseSoft:
            out.write('{} 0\n'.format(cnf))
        for cnf in clauseHard:
            out.write('{} {} 0\n'.format(hardWeight, cnf))
    
    a = time.time()
    command = '{} {}'.format(csp_solver_path, outfile)
    proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = proc.communicate()
    b = time.time()

    variables = output.decode().split('\n')[-2][2:].split(' ')
    O = np.empty((n,m), dtype=np.int8)
    numVar = 0
    for i in range(n):
        for j in range(m):
            if matrix[i,j] == 0:
                if '-' in variables[numVar]:
                    O[i,j] = 0
                else:
                    O[i,j] = 1
            elif matrix[i,j] == 1:
                if '-' in variables[numVar]:
                    O[i,j] = 0
                else:
                    O[i,j] = 1
            elif matrix[i,j] == 2:
                if '-' in variables[numVar]:
                    O[i,j] = 0
                else:
                    O[i,j] = 1
            numVar += 1
    return O, count_flips(matrix, matrix.shape[1]*[0], O), b-a


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
                if D[r,p] == 1 and D[r,j] == 1:
                    oneone += 1
                    if r == i:
                        founded = True
                if D[r,p] == 0 and D[r,j] == 1:
                    zeroone += 1
                    if r == i:
                        founded = True
                if D[r,p] == 1 and D[r,j] == 0:
                    onezero += 1
                    if r == i:
                        founded = True
            if founded:
                total += oneone*zeroone*onezero
        return total

    violations = {}
    for r in range(D.shape[0]):
        for p in range(D.shape[1]):
            if D[r,p] == 0:
                violations[(r,p)] = calc_how_many_violations_are_in(D, r, p)

    for x in sorted(violations.items(), key=operator.itemgetter(1), reverse=True)[:10]:
        print(x[0], '(entry={}): how many gametes'.format(D[x[0]]), x[1])

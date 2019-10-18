from Utils.const import *
from Utils.interfaces import *


def PhISCS_B_helper(matrix):
    rc2 = RC2(WCNF())
    n, m = matrix.shape
    fnWeight = 1
    fpWeight = 10

    Y = np.empty((n, m), dtype=np.int64)
    numVarY = 0
    mapY2ij = {}
    for i in range(n):
        for j in range(m):
            numVarY += 1
            mapY2ij[numVarY] = (i, j)
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
                rc2.add_clause([-X[i, j]], weight=fnWeight)
                rc2.add_clause([-X[i, j], Y[i, j]])
                rc2.add_clause([X[i, j], -Y[i, j]])
            elif matrix[i, j] == 1:
                rc2.add_clause([-X[i, j]], weight=fpWeight)
                rc2.add_clause([X[i, j], Y[i, j]])
                rc2.add_clause([-X[i, j], -Y[i, j]])

    for i in range(n):
        for p in range(m):
            for q in range(p, m):
                rc2.add_clause([-Y[i, p], -Y[i, q], B[p, q, 1, 1]])
                rc2.add_clause([Y[i, p], -Y[i, q], B[p, q, 0, 1]])
                rc2.add_clause([-Y[i, p], Y[i, q], B[p, q, 1, 0]])
                rc2.add_clause([-B[p, q, 0, 1], -B[p, q, 1, 0], -B[p, q, 1, 1]])

    variables = rc2.compute()
    return sum(i > 0 for i in variables[n * m : 2 * n * m])


class StaticCSPBounding(BoundingAlgAbstract):
    def __init__(self, splitInto=2):
        self.matrix = None
        self.n = None
        self.m = None
        self.splitInto = splitInto

    def reset(self, matrix):
        self.matrix = matrix
        self.n = self.matrix.shape[0]
        self.m = self.matrix.shape[1]

    def get_name(self):
        return type(self).__name__ + "_" + str(self.splitInto)

    def get_bound(self, delta):
        # https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
        bound = 0
        I = np.array(self.matrix + delta)
        blocks = np.array_split(I, self.splitInto, axis=1)
        for block in blocks:
            bound += PhISCS_B_helper(block)
        return bound + delta.count_nonzero()


class SemiDynamicCSPBounding(BoundingAlgAbstract):
    def __init__(self, splitInto=2):
        self.matrix = None
        self.n = None
        self.m = None
        self.times = None
        self.blockIndices = None
        self.splitInto = splitInto
        self.models = []
        self.yVars = []

    def reset(self, matrix):
        self.matrix = matrix
        self.times = {"modelPreperationTime": 0, "optimizationTime": 0}
        self.n = self.matrix.shape[0]
        self.m = self.matrix.shape[1]
        self.blockIndices = [list(x) for x in np.array_split(range(self.m), self.splitInto)]
        modelTime = time.time()
        for block in self._get_blocks(self.matrix):
            model, yVar = self._make_model(block)
            self.models.append(model)
            self.yVars.append(yVar)
        modelTime = time.time() - modelTime
        self.times["modelPreperationTime"] += modelTime
        # optTime = time.time()
        # for model in self.models:
        #     variables = model.compute()
        # optTime = time.time() - optTime
        # self.times["optimizationTime"] += optTime

    def _get_blocks(self, I):
        return [I[:, i] for i in self.blockIndices]

    def _make_model(self, subMatrix):
        wcnf = WCNF()
        n = subMatrix.shape[0]
        m = subMatrix.shape[1]
        fnWeight = 1
        fpWeight = 10

        Y = np.empty((n, m), dtype=int)
        numVarY = 0
        for i in range(n):
            for j in range(m):
                numVarY += 1
                Y[i, j] = numVarY

        X = np.empty((n, m), dtype=int)
        numVarX = 0
        for i in range(n):
            for j in range(m):
                numVarX += 1
                X[i, j] = numVarY + numVarX

        B = np.empty((m, m, 2, 2), dtype=int)
        numVarB = 0
        for p in range(m):
            for q in range(m):
                for i in range(2):
                    for j in range(2):
                        numVarB += 1
                        B[p, q, i, j] = numVarY + numVarX + numVarB

        for i in range(n):
            for j in range(m):
                if subMatrix[i, j] == 0:
                    wcnf.append([-X[i, j].item()], weight=fnWeight)
                    wcnf.append([-X[i, j].item(), Y[i, j].item()])
                    wcnf.append([X[i, j].item(), -Y[i, j].item()])
                elif subMatrix[i, j] == 1:
                    wcnf.append([-X[i, j].item()], weight=fpWeight)
                    wcnf.append([X[i, j].item(), Y[i, j].item()])
                    wcnf.append([-X[i, j].item(), -Y[i, j].item()])

        for i in range(n):
            for p in range(m):
                for q in range(p, m):
                    wcnf.append([-Y[i, p].item(), -Y[i, q].item(), B[p, q, 1, 1].item()])
                    wcnf.append([Y[i, p].item(), -Y[i, q].item(), B[p, q, 0, 1].item()])
                    wcnf.append([-Y[i, p].item(), Y[i, q].item(), B[p, q, 1, 0].item()])
                    wcnf.append([-B[p, q, 0, 1].item(), -B[p, q, 1, 0].item(), -B[p, q, 1, 1].item()])
        return wcnf, Y

    def get_name(self):
        return type(self).__name__ + "_" + str(self.splitInto)

    def get_bound(self, delta):
        modelTime = time.time()
        cx = delta.tocoo()
        # print('-------------', cx.row, cx.col, cx.data)
        models = []
        for model in self.models:
            new_model = model.copy()
            models.append(new_model)
        for i, j, v in zip(cx.row, cx.col, cx.data):
            whichBlock = -1
            indexInBlock = -1
            for k in range(self.splitInto):
                if j in self.blockIndices[k]:
                    whichBlock = k
                    indexInBlock = self.blockIndices[k].index(j)
                    break
            models[whichBlock].append([self.yVars[whichBlock][i, indexInBlock].item()])
        modelTime = time.time() - modelTime
        self.times["modelPreperationTime"] += modelTime

        bound = 0
        optTime = time.time()
        for j in range(self.splitInto):
            model = models[j]
            rc2 = RC2(model)
            variables = rc2.compute()
            m = len(self.blockIndices[j])
            bound += sum(i > 0 for i in variables[self.n * m : 2 * self.n * m])
        optTime = time.time() - optTime
        self.times["optimizationTime"] += optTime
        return bound


if __name__ == "__main__":

    noisy = np.array(
        [
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    delta = sp.lil_matrix((noisy.shape), dtype=int)

    algo1 = StaticCSPBounding(4)
    algo1.reset(noisy)
    print(algo1.get_bound(delta))

    algo2 = SemiDynamicCSPBounding(4)
    algo2.reset(noisy)
    print(algo2.get_bound(delta))
    print(algo2.times)

    delta[0, 0] = 1
    print(algo2.get_bound(delta))
    delta[0, 5] = 1
    print(algo2.get_bound(delta))
    delta[0, 9] = 1
    print(algo2.get_bound(delta))

    print(algo1.get_bound(delta))

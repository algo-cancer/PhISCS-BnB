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

    def getName(self):
        return type(self).__name__ + "_" + str(self.splitInto)

    def getBound(self, delta):
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
        for block in self._getBlocks(self.matrix):
            model, yVar = self._makeModel(block)
            self.models.append(model)
            self.yVars.append(yVar)
        modelTime = time.time() - modelTime
        self.times["modelPreperationTime"] += modelTime
        # optTime = time.time()
        # for model in self.models:
        #     variables = model.compute()
        # optTime = time.time() - optTime
        # self.times["optimizationTime"] += optTime

    def _getBlocks(self, I):
        return [I[:, i] for i in self.blockIndices]

    def _makeModel(self, subMatrix):
        rc2 = RC2(WCNF())
        n = subMatrix.shape[0]
        m = subMatrix.shape[1]
        fnWeight = 1
        fpWeight = 10

        Y = np.empty((n, m), dtype=np.int64)
        numVarY = 0
        for i in range(n):
            for j in range(m):
                numVarY += 1
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
                if subMatrix[i, j] == 0:
                    rc2.add_clause([-X[i, j]], weight=fnWeight)
                    rc2.add_clause([-X[i, j], Y[i, j]])
                    rc2.add_clause([X[i, j], -Y[i, j]])
                elif subMatrix[i, j] == 1:
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
        return rc2, Y

    def getName(self):
        return type(self).__name__ + "_" + str(self.splitInto)

    def getBound(self, delta):
        modelTime = time.time()
        cx = delta.tocoo()
        models = []
        for model in self.models:
            models.append(copy.copy(model))
        for i, j, v in zip(cx.row, cx.col, cx.data):
            whichBlock = -1
            indexInBlock = -1
            for k in range(self.splitInto):
                if j in self.blockIndices[k]:
                    whichBlock = k
                    indexInBlock = self.blockIndices[k].index(j)
                    break
            models[whichBlock].add_clause([self.yVars[whichBlock][i, indexInBlock]])
        modelTime = time.time() - modelTime
        self.times["modelPreperationTime"] += modelTime

        bound = 0
        optTime = time.time()
        for j in range(self.splitInto):
            model = models[j]
            variables = model.compute()
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
    delta[0, 0] = 1
    delta[0, 5] = 1
    delta[0, 9] = 1

    algo = StaticCSPBounding(4)
    algo.reset(noisy)
    print(algo.getBound(delta))

    algo = SemiDynamicCSPBounding(4)
    algo.reset(noisy)
    print(algo.getBound(delta))
    print(algo.times)

from utils.util import *


def simple_alg(x, mx_iter = 50):
    sol = x.copy()
    for ind in range(mx_iter):
        icf, col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(sol)
        # print(icf, col_pair)
        if icf:
            return True, sol
        col1 = sol[:, col_pair[0]]
        col2 = sol[:, col_pair[1]]
        rows01 = np.nonzero(np.logical_and(col1 == 0, col2 == 1))[0]
        # print(ind, len(rows01))
        sol[rows01, col_pair[0]] = 1
    return False, sol


if __name__ == '__main__':
    x = np.array([[0, 1, 0, 1, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1]])

    # x = np.array([[0, 1],
    #               [1, 0],
    #               [1, 1]])
    solved, sol = simple_alg(x)
    print(solved)
    print(sol)

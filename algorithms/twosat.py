from utils.const import *
from utils.util import *
from algorithms.PhISCS import PhISCS_B
rec_num = 0


def twosat_solver(matrix, cluster_rows=False, cluster_cols=False, only_descendant_rows=False,
                  na_value=None, leave_nas_if_zero=False, return_lb=False, heuristic_setting=None,
                  n_levels=2, eps=0, compact_formulation=False):
    """
    This algorithm is based on iterative usage of weighted 2-sat solver.
    It runs in polynomial time (TODO complexity) and is theoretically guaranteed to give an upper bound
    :param threshold:
    :param matrix:
    :return:
    """
    global rec_num
    rec_num += 1
    print(rec_num)
    assert is_na_set_correctly(matrix, na_value)
    assert not cluster_rows, "Not implemented yet"
    assert not cluster_cols, "Not implemented yet"
    assert not only_descendant_rows, "Not implemented yet"
    # next(plg)
    model_time = 0
    opt_time = 0
    start_time = time.time()
    # print_line()
    # next(plg)

    return_value = make_constraints_np_matrix(matrix, n_levels=n_levels, na_value=na_value,
                                              compact_formulation=compact_formulation)
    # next(plg)
    # print_line()
    model_time += time.time() - start_time
    # for i in range(len(return_value.hard_constraints)):
    #     print(len(return_value.hard_constraints[i]))
    # print_line()

    # next(plg)
    F, map_f2ij, zero_vars, na_vars, hard_constraints, col_pair = \
        return_value.F, return_value.map_f2ij, return_value.zero_vars, return_value.na_vars,\
        return_value.hard_constraints, return_value.col_pair

    zeroed_matrix = matrix.copy()
    zeroed_matrix[zeroed_matrix == na_value] = 0
    # exit(0)
    # next(plg)
    if col_pair is not None:
        icf = False
    elif return_value.complete_version:
        icf = True
    else:
        icf = None

    final_output = None
    lower_bound = 0
    if icf:
        final_output, total_time = matrix.copy(), 0
    else:
        start_time = time.time()
        # next(plg)
        rc2 = make_twosat_model_from_np(hard_constraints, F, zero_vars, na_vars, eps, heuristic_setting,
                                        compact_formulation=compact_formulation)
        model_time += time.time() - start_time
        # next(plg)

        print(now())
        a = time.time()
        variables = rc2.compute()
        b = time.time()
        # next(plg)
        opt_time += b - a
        output_matrix = matrix.copy()
        output_matrix = output_matrix.astype(np.int8)

        for var_ind in range(len(variables)):
            if 0 < variables[var_ind] and variables[var_ind] in map_f2ij:  # if 0 or 2 make it one
                output_matrix[map_f2ij[variables[var_ind]]] = 1
                if matrix[map_f2ij[variables[var_ind]]] != na_value:
                    lower_bound += 1
        # I don't change 2s to 0s here keep them 2 for next time

        # For recursion I set off all sparsification parameters
        # Also I want na->0 to stay na for the recursion regardless of original input for leave_nas_if_zero
        # I am also not passing eps here to wrap up the recursion soon


        Orec, rec_model_time, rec_opt_time = twosat_solver(output_matrix, na_value=na_value,
                                      heuristic_setting=None, n_levels=n_levels, leave_nas_if_zero=True,
                                      compact_formulation=compact_formulation)
        # next(plg)
        model_time += rec_model_time
        opt_time += rec_opt_time

        if not leave_nas_if_zero:
            Orec[Orec == na_value] = 0
        final_output = Orec

    # print("lower_bound=", lower_bound, len(map_f2ij))
    if return_lb:
        return final_output, model_time, opt_time, lower_bound
    else:
        return final_output, model_time, opt_time


def make_twosat_model(matrix, threshold=0, coloring=None, na_value=None, eps=None, probability_threshold=None,
                      fn_rate=None, use_heuristics=False):
    global setting
    assert is_na_set_correctly(matrix, na_value)
    assert (probability_threshold is None) == (fn_rate is None)

    complete_version = coloring is None and probability_threshold is None and fn_rate is None
    if eps is None:
        eps = 1 / (matrix.shape[0] + matrix.shape[1])
    hard_cnst_num = 0
    soft_cnst_num = 0
    if use_heuristics:
        rc2 = RC2(WCNF(), adapt=setting[0], exhaust=setting[1], incr=setting[2], minz=setting[3], trim=setting[4])
        # rc2 = RC2(WCNF(), adapt=True, exhaust=True, incr=False, minz=True, trim=True)
    else:
        rc2 = RC2(WCNF())

    n, m = matrix.shape

    # dictionary for lazy calculation of decadence:
    descendent_dict = dict()

    # variables for each zero
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

        # if p % 500 == 0:
        #     print_line()
        #     print(p)
        for q in range(p + 1, m):
            if np.any(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 1)):  # p and q has intersection
                # print_line()
                r01 = np.nonzero(np.logical_and(zero_or_na(matrix[:, p], na_value=na_value), matrix[:, q] == 1))[0]
                r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, zero_or_na(matrix[:, q], na_value=na_value)))[0]
                cost = min(len(r01), len(r10))
                if cost > pair_cost:  # keep best pair to return as auxiliary info
                    col_pair = (p, q)
                    pair_cost = cost

                if coloring is not None and coloring[p] != coloring[q]:
                    continue
                direct_formulation_cost = len(r01) * len(r10)
                indirect_formulation_cost = len(r01) + len(r10)
                if direct_formulation_cost * threshold <= indirect_formulation_cost:  # use direct formulation
                    for a, b in itertools.product(r01, r10):
                        for x, y in [(a, p), (b, q)]:  # make sure the variables for this are made
                            if F[x, y] < 0:
                                num_var_F += 1
                                map_f2ij[num_var_F + num_var_B] = (x, y)
                                F[x, y] = num_var_F + num_var_B
                                if matrix[x, y] == 0:  # So as not to add weight for N/A s
                                    w = 1
                                else:
                                    w = eps
                                if w > 0:
                                    rc2.add_clause([-F[x, y]], weight=w)
                                    soft_cnst_num += 1

                        if probability_threshold is not None and fn_rate is not None:
                            assert False, "should have not come here"
                            if (a, b) not in descendent_dict:
                                n_ones = np.sum(np.logical_and(matrix[a] == 1, matrix[b] == 1))
                                n_switched1 = np.sum(np.logical_and(matrix[a] == 0, matrix[b] == 1))
                                n_switched2 = np.sum(np.logical_and(matrix[a] == 1, matrix[b] == 0))
                                null_probability1 = null_prob(n_switched1, n_ones, fn_rate)
                                null_probability2 = null_prob(n_switched2, n_ones, fn_rate)
                                descendent_dict[(a, b)] = null_probability1 > probability_threshold \
                                                          or null_probability2 > probability_threshold

                            if not descendent_dict[(a, b)]:
                                continue  # ignore this constraint if there rows are not decedent of each other
                                # todo: do this for the indirect formulation
                        rc2.add_clause([F[a, p], F[b, q]])  # at least one of them should be flipped
                        hard_cnst_num += 1
                        # print_line()
                else:  # use indirect formulation
                    assert False, "should have not come here"
                    if cost > 0:
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
    # print("hard_cnst_num, soft_cnst_num =", hard_cnst_num, soft_cnst_num)

    if (not complete_version) and hard_cnst_num < 10000 and soft_cnst_num > 0:
        assert False, "should have not come here"

        print("cancel constraint elemination!")  # todo: when canceling just use optimal physics_b
        return make_twosat_model(
            matrix=matrix,
            threshold=threshold,
            coloring=None,
            na_value=na_value,
            eps=eps,
            probability_threshold=None,
            fn_rate=None,
        )

    return rc2, col_pair, map_f2ij, map_b2pq


def make_clustered_2sat_model(
        matrix, row_partitioning=None, col_partitioning=None, na_value=2, eps=None):
    # print(len(row_partitioning), len(col_partitioning))
    if eps is None:
        eps = 1 / (matrix.shape[0] + matrix.shape[1])
    print(len(row_partitioning), len(col_partitioning))

    if row_partitioning is None:
        row_partitioning = [list(range(matrix.shape[0])), ]

    row_same_partition = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.int8)

    for row_par in row_partitioning:
        row_same_partition[np.ix_(row_par, row_par)] = 1

    if col_partitioning is None:
        col_partitioning = [list(range(matrix.shape[1])), ]

    hard_cnst_num = 0
    soft_cnst_num = 0
    rc2 = RC2(WCNF(),
              # adapt=True,
              # exhaust=True,
              # incr=False,
              # minz=True,
              # trim=True,
              )
    n, m = matrix.shape

    # variables for each zero
    F = - np.ones((n, m), dtype=np.int64)
    num_var_F = 0
    map_f2ij = {}

    num_var_B = 0  # for future

    col_pair = None
    pair_cost = 0
    # print(col_partitioning, row_same_partition)
    for col_par in col_partitioning:
        for p in col_par:
            for q in col_par:
                if p < q and np.any(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 1)):  # p and q has intersection
                    r01 = np.nonzero(np.logical_and(zero_or_na(matrix[:, p]), matrix[:, q] == 1))[0]
                    r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, zero_or_na(matrix[:, q])))[0]
                    cost = min(len(r01), len(r10))
                    if cost > pair_cost:  # keep best pair to return as auxiliary info
                        col_pair = (p, q)
                        pair_cost = cost

                    for a, b in itertools.product(r01, r10):
                        if row_same_partition[a, b]:
                            for x, y in [(a, p), (b, q)]:
                                if F[x, y] < 0:  # make sure the variables for this are made
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
                            rc2.add_clause([F[a, p], F[b, q]])  # at least one of them should be flipped
                            hard_cnst_num += 1
    # print(hard_cnst_num, soft_cnst_num)
    return rc2, col_pair, map_f2ij



#$$$$$$$$$$$$$$$$$$$444
def make_twosat_model_from_np(constraints, F, zero_vars, na_vars,  eps=None,
                              heuristic_setting=None, compact_formulation=True):

    if eps is None:
        eps = 1 / (len(zero_vars) + len(na_vars))

    if heuristic_setting is None:
        rc2 = RC2(WCNF())
    else:
        assert len(heuristic_setting) == 5
        rc2 = RC2(WCNF(),
                  adapt=heuristic_setting[0],
                  exhaust=heuristic_setting[1],
                  incr=heuristic_setting[2],
                  minz=heuristic_setting[3],
                  trim=heuristic_setting[4])

    if not compact_formulation:
        # hard constraints Z_a,p or Z_b,q
        for constr_ind in range(constraints[0].shape[0]):
            constraint = constraints[0][constr_ind]
            a, p, b, q = constraint.flat
            # print(constraint, F.shape)
            # print(a, p, b, q)
            rc2.add_clause([F[a, p], F[b, q]])
        if len(constraints) >= 2:
            # hard constraints Z_a,p or Z_b,q or -Z_c,d
            for constr_ind in range(constraints[1].shape[0]):
                constraint = constraints[1][constr_ind]
                a, p, b, q, c, d = constraint.flat
                # print(a, p, b, q, c, d)
                rc2.add_clause([F[a, p], F[b, q], -F[c, d]])
    else:
        # hard constraints Z_a,p or (sign) b_pq
        for constr_ind in range(constraints[0].shape[0]):
            constraint = constraints[0][constr_ind]
            row, col, b_pq, sign = constraint.flat
            rc2.add_clause([F[row, col], sign*b_pq])
        if len(constraints) >= 2:
            # hard constraints Z_a,p or Z_b,q or -Z_c,d
            for constr_ind in range(constraints[1].shape[0]):
                constraint = constraints[1][constr_ind]
                row, col, c_pq0, c_pq1 = constraint.flat
                # if Z_rc is True at least one of p, q should become active
                # E.g., c_pq0 be False
                rc2.add_clause([- F[row, col], - c_pq0, - c_pq1])
                # if c_pq0 is False then Z_rc has to be flipped
                rc2.add_clause([F[row, col], c_pq0])



    # soft constraints for zero variables
    for var in zero_vars:
        rc2.add_clause([-var], weight=1)

    if eps > 0:
        # soft constraints for zero variables
        for var in na_vars:
            rc2.add_clause([-var], weight=eps)

    return rc2



def calculate_column_intersections(matrix, for_loop=False, row_by_row=False):
    ret = np.empty((matrix.shape[1], matrix.shape[1]), dtype=np.bool)
    mask_1 = matrix == 1

    if for_loop:
        for p in range(matrix.shape[1]):
            # even though the diagonals are not necessary, I keep it for ease of debugging
            for q in range(p, matrix.shape[1]):
                ret[p, q] = np.any(np.logical_and(mask_1[:, p], mask_1[:, q]))
                ret[q, p] = ret[p, q]
    elif row_by_row:
        ret[:, :] = 0
        for r in range(matrix.shape[0]):
            one_columns = mask_1[r]
            ret[np.ix_(one_columns, one_columns)] = True
    return ret


def make_sure_variable_exists(memory_matrix, row, col, num_var_F, map_f2ij, var_list, na_value):
    if memory_matrix[row, col] < 0:
        num_var_F += 1
        map_f2ij[num_var_F] = (row, col)
        memory_matrix[row, col] = num_var_F
        var_list.append(num_var_F)
    return num_var_F


def make_constraints_np_matrix(matrix, constraints=None, n_levels=2, na_value=None,
                               row_coloring=None, col_coloring=None,
                               probability_threshold=None, fn_rate=None,
                               column_intersection = None, compact_formulation = True):
    """
    Returns a "C x 2 x 2" matrix where C is the number of extracted constraints each constraints is of the form:
    ((r1, c1), (r2, c2)) and correspond to Z_{r1, c1} or Z{r2, c2}
    :param matrix: A binary matrix cellsXmutations
    :param constraints: If not None instead of evaluating the whole matrix it will only look at potential constraints
    :param level: The type of constraints to add
    :param na_value:
    :param row_coloring: Only constraints that has the same row coloring will be used
    :param col_coloring: Only constraints that has the same column coloring will be used
    :param probability_threshold:
    :param fn_rate:
    :return:
    """
    # todo: Take decendence analysis out of here?
    # todo: how to reuse constraints input
    from collections import namedtuple
    assert is_na_set_correctly(matrix, na_value)
    assert (probability_threshold is None) == (fn_rate is None)
    descendance_analysis = probability_threshold is not None
    assert 1 <= n_levels <= 2, "not implemented yet"

    # means none of scarification ideas have been used
    complete_version = all_None(row_coloring, col_coloring, probability_threshold, fn_rate)

    soft_cnst_num = 0
    hard_constraints = [[] for _ in range(n_levels)]  # an empty list each level
    if descendance_analysis:
        # dictionary for lazy calculation of decadence:
        descendent_dict = dict()

    # variables for each zero
    F = - np.ones(matrix.shape, dtype=np.int64)
    num_var_F = 0
    map_f2ij = dict()
    zero_vars = list()
    na_vars = list()
    if compact_formulation:
        B_vars_offset = matrix.shape[0] * matrix.shape[1] + 1
        num_var_B = 0
        map_b2ij = dict()
        if n_levels >= 2:
            C_vars_offset = B_vars_offset + matrix.shape[1] * matrix.shape[1] + 1
            num_var_C = 0
            map_c2ij = dict()


    col_pair = None
    pair_cost = 0

    if column_intersection is None:
        column_intersection = calculate_column_intersections(matrix, row_by_row=True)
        # column_intersection = calculate_column_intersections(matrix, for_loop=True)
    for p in range(matrix.shape[1]):
        for q in range(p + 1, matrix.shape[1]):
            if column_intersection[p, q]:  # p and q has intersection
                # todo: check col_coloring here
                r01 = np.nonzero(np.logical_and(zero_or_na(matrix[:, p], na_value=na_value), matrix[:, q] == 1))[0]
                r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, zero_or_na(matrix[:, q], na_value=na_value)))[0]
                cost = min(len(r01), len(r10))
                if cost > pair_cost:  # keep best pair to return as auxiliary info
                    # print("------------", cost, (p, q), len(r01), len(r10), column_intersection[p, q])
                    col_pair = (p, q)
                    pair_cost = cost
                if cost > 0:  # don't do anything if one of r01 or r10 is empty
                    if not compact_formulation:  # len(r01) * len(r10) many constraints will be added
                        for a, b in itertools.product(r01, r10):
                            # todo: check row_coloring
                            for row, col in [(a, p), (b, q)]:  # make sure the variables for this are made
                                var_list = zero_vars if matrix[row, col] == 0 else na_vars
                                num_var_F = make_sure_variable_exists(F, row, col,
                                                                      num_var_F, map_f2ij, var_list, na_value)
                            hard_constraints[0].append([[a, p], [b, q]])  # at least one of them should be flipped
                    else:  # compact formulation: (r01 + r10) number of new constraints will be added
                        # define new B variable
                        b_pq = B_vars_offset + num_var_B
                        num_var_B += 1
                        for row_list, col, sign in zip((r01, r10), (p, q), (1, -1)):
                            for row in row_list:
                                var_list = zero_vars if matrix[row, col] == 0 else na_vars
                                num_var_F = make_sure_variable_exists(F, row, col,
                                                                  num_var_F, map_f2ij, var_list, na_value)
                                hard_constraints[0].append([row, col, b_pq, sign])
                                # this will be translated to (Z_ap or (sign)B_pq)
            elif n_levels >= 2:
                r01 = np.nonzero(np.logical_and(zero_or_na(matrix[:, p], na_value=na_value), matrix[:, q] == 1))[0]
                r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, zero_or_na(matrix[:, q], na_value=na_value)))[0]
                cost = min(len(r01), len(r10))
                if cost > 0:  # don't do anything if one of r01 or r10 is empty
                    if not compact_formulation:
                        # len(r01) * len(r10) * (len(r01) * len(r10)) many constraints will be added
                        x = np.empty((r01.shape[0] + r10.shape[0], 2), dtype = np.int)
                        x[:len(r01), 0] = r01
                        x[:len(r01), 1] = p
                        x[-len(r10):, 0] = r10
                        x[-len(r10):, 1] = q

                        for a, b, ind in itertools.product(r01, r10, range(x.shape[0])):
                            for row, col in [(a, p), (b, q), (x[ind, 0], x[ind, 1])]:  # make sure the variables for this are made
                                # print(row, col)
                                var_list = zero_vars if matrix[row, col] == 0 else na_vars
                                num_var_F = make_sure_variable_exists(F, row, col,
                                                                      num_var_F, map_f2ij, var_list, na_value)
                            row = [[a, p], [b, q], [x[ind, 0], x[ind, 1]]]
                            if not np.array_equal(row[0], row[2]) and not np.array_equal(row[1], row[2]):
                                hard_constraints[1].append([[a, p], [b, q], [x[ind, 0], x[ind, 1]]])
                    else: #  if compact_formulation: 2(r01 + r10) will be added
                        # define two new C variable
                        c_pq0 = C_vars_offset + num_var_C
                        num_var_C += 1
                        c_pq1 = C_vars_offset + num_var_C
                        num_var_C += 1
                        for row_list, col, sign in zip((r01, r10), (p, q), (1, -1)):
                            for row in row_list:
                                var_list = zero_vars if matrix[row, col] == 0 else na_vars
                                num_var_F = make_sure_variable_exists(F, row, col,
                                                                  num_var_F, map_f2ij, var_list, na_value)
                                if sign == 1:
                                    hard_constraints[1].append([row, col, c_pq0, c_pq1])
                                    # this will be translated to (~Z_ap or ~c_pq0 or ~c_pq1)
                                    # and (Z_ap or c_pq0)
                                else:
                                    hard_constraints[1].append([row, col, c_pq1, c_pq0])
                                    # this will be translated to (~Z_ap or ~c_pq0 or ~c_pq1) (the same)
                                    # and (Z_ap or c_pq1) (different)


    # todo: when using this make sure to put an if to say if the model is small and
    return_type = namedtuple("ReturnType", "F map_f2ij zero_vars na_vars hard_constraints col_pair complete_version")
    for ind in range(n_levels):
        hard_constraints[ind] = np.array(hard_constraints[ind], dtype=np.int)
    return return_type(F, map_f2ij, zero_vars, na_vars, hard_constraints, col_pair, complete_version)





######################################################################33
# if descendance_analysis:  # todo:  descendance_analysis
#     assert False, "should have not come here"
#     if (a, b) not in descendent_dict:
#         n_ones = np.sum(np.logical_and(matrix[a] == 1, matrix[b] == 1))
#         n_switched1 = np.sum(np.logical_and(matrix[a] == 0, matrix[b] == 1))
#         n_switched2 = np.sum(np.logical_and(matrix[a] == 1, matrix[b] == 0))
#         null_probability1 = null_prob(n_switched1, n_ones, fn_rate)
#         null_probability2 = null_prob(n_switched2, n_ones, fn_rate)
#         descendent_dict[(a, b)] = null_probability1 > probability_threshold \
#                                   or null_probability2 > probability_threshold
#
#     if not descendent_dict[(a, b)]:
#         continue  # ignore this constraint if there rows are not decedent of each other
######################################################################33



#
# if __name__ == '__main__':
#     # filename = "simNo_2-s_4-m_50-n_50-k_50.SC.noisy"
#     # folder_path = "Clean"
#     #
#     filename = "SCTrioSeq_selected_genes.SC"
#     folder_path = "Data/real"
#     x = read_matrix_from_file(file_name=filename, folder_path=folder_path)
#     x = x[:, :]
#     na_value = infer_na_value(x)
#     x[x==na_value]=0
#     na_value=None
#
#     # x = np.array([[1, 0, 3],
#     #               [0, 1, 3],
#     #               [1, 1, 3]
#     #               ])
#
#     print(x.shape)
#     result = twosat_solver(x, 0, na_value=na_value, leave_nas_if_zero=False, return_lb=True, use_heuristics=False)
#     print(np.unique(result[0]))
#     flips_0_1, flips_1_0, flips_na_0, flips_na_1 = count_flips(I=x, sol_Y=result[0], na_value=na_value)
#     print(flips_0_1, flips_1_0, flips_na_0, flips_na_1, result[1:])

setting = None
plg = print_line_iter()

if __name__ == '__main__':
    print('hello!')
    print(now())
    # im = np.array([[1, 2, 1],
    #                [0, 1, 0],
    #                [1, 1, 0]])
    #
    # im = np.array([[0, 1, 0],
    #                [0, 1, 0],
    #                [1, 0, 1],
    #                [1, 0, 1],
    #                [1, 1, 0]])
    # next(plg)
    folderpath = "Data/real"
    filename = "SCTrioSeq_cancer_genes_LN.SC"
    filename = "SCTrioSeq_cancer_genes.SC"
    im = read_matrix_from_file(file_name=filename, folder_path=folderpath)
    im = im[:40, :40]

    # c1 = calculate_column_intersections(im, for_loop=True)
    # c2 = calculate_column_intersections(im, row_by_row=True)
    # co = c1 != c2
    # print(c1)
    # print(c2)
    # print(np.sum(co))
    # print(c1[5, 9])
    # print(im[:, [5, 9]])
    # exit(0)

    # im = im[:, :80]
    im[im == 3] = 0
    # print(im)
    # print(repr(im))
    print(im.shape)
    na_value = 3
    args = {
        "matrix": im,
        "leave_nas_if_zero" : False,
        "return_lb": True,
        "heuristic_setting": [True, True, False, True, True],
        # "heuristic_setting": None,
        "n_levels": 2,
        "eps": 0,
        "compact_formulation": True,
        "na_value": na_value,
        "alpha": 0.0001,
        "beta": 0.9,
    }
    # result = timed_run(PhISCS_B, args=args, time_limit=1200)
    result = timed_run(twosat_solver, args=args, time_limit=600)
    print(result["input"])
    output_matrix = result["output"][0]
    nf = count_flips(im, sol_K=[0] * im.shape[1], sol_Y=output_matrix, na_value=na_value)
    print("nf=", nf)
    print(result["output"][1:])
    print(f"NA value: {na_value}")
    print(f"#Zeros: {len(np.where(im == 0)[0])}")
    print(f"#Ones: {len(np.where(im == 1)[0])}")
    print(f"#NAs: {len(np.where(im == na_value)[0])}")
    print("rec#=", rec_num)
    # def make_constraints_np_matrix(matrix, constraints=None, n_levels=2, na_value=None,
    #                                row_coloring=None, col_coloring=None,
    #                                probability_threshold=None, fn_rate=None,
    #                                column_intersection=None, compact_formulation=True):

    # next(plg)
    # rt = make_constraints_np_matrix(im, n_levels=2, compact_formulation=True, na_value=2)
    # next(plg)
    # rc2 = make_twosat_model_from_np(rt.hard_constraints, rt.F, rt.zero_vars, rt.na_vars,
    #                                 eps=None, heuristic_setting=None)
    # print(rt)
    # print(len(rt.hard_constraints))
    # print(len(rt.hard_constraints[0]))
    # print(len(rt.hard_constraints[1]))
    # next(plg)
    exit(0)
    # def twosat_solver(matrix, threshold=0, cluster_rows=False, cluster_cols=False, only_descendant_rows=False,
    #                   na_value=None, leave_nas_if_zero=False, return_lb=False, heuristic_setting=None, level=0, eps=0):

    result = twosat_solver(im, leave_nas_if_zero=True, return_lb=True, n_levels=2, compact_formulation=True)
    next(plg)
    print(result[1], result[2], result[3])
    nnn = count_nf(result[0], im)
    print("finale nf=", nnn)
    next(plg)
    # result = make_constraints_np_matrix(im)
    # print(result.hard_constraints.shape)
    exit(0)
    exit(0)
    # make_twosat_model(im)
    # ci1 = calculate_column_intersections(im, for_loop=True)
    next(plg)
    # ci2 = calculate_column_intersections(im, row_by_row=True)
    # result = make_constraints_np_matrix(im, column_intersection=ci)
    next(plg)
    # print(np.all(ci1 == ci2))
    # print(result)
    exit(0)

    input_matrix = df_input.values
    # matrix = matrix[:, :]

    next(plg)
    result = make_twosat_model(input_matrix, use_heuristics=False)
    next(plg)
    exit(0)

    a = time.time()
    result = twosat_solver(matrix, use_heuristics=False, return_lb=True)
    a = time.time() - a
    next(plg)
    nf = count_nf(result[0], matrix)
    next(plg)
    print(a, nf, result[-1])
    exit(0)
    for setting in itertools.product([False, True], repeat=5):
        a = time.time()
        result = twosat_solver(matrix, use_heuristics=True, return_lb=True)
        a = time.time() - a
        print(setting, a)

'''
Algorithms for UCM Optimization
A: MIP Formulation
B: adxOpt1_products
C: adxOpt2_sets
D: Brute Search
'''

import numpy as np
import time
import collections
from itertools import combinations
from itertools import chain
import docplex.mp.model as cpx
from synthetic_models.utils import set_char_from_ast, ast_from_set_char
import logging

logger = logging.getLogger(__name__)


def init_optim_algorithms():
    global optim_methods_src
    optim_methods_src = {
        'revenue-ordered': ucm_revenue_ordered,
        'mixed-ip': ucm_mixed_ip,
        'adxopt1': ucm_adxopt1_products,
        'adxopt2': ucm_adxopt2_sets,
        'brute-force': ucm_brute_force_search
    }


def run_ucm_optimization(algorithm, num_prods, C, rcm_model, meta):
    init_optim_algorithms()
    global optim_methods_src
    optim_function = optim_methods_src[algorithm]
    maxRev, maxSet, timeTaken = optim_function(num_prods, C, rcm_model, meta)
    logger.info(f"Algorithm: {algorithm},MaxRev: {maxRev},MaxSet: {maxSet}, TimeTaken:{str(timeTaken)}")
    return {'max_revenue': maxRev, 'max_set': maxSet, 'time_taken': timeTaken}


# ====================UCM Revenue Ordered Assortments ====================================
def ucm_revenue_ordered_old(num_prods, C_old, ucm, meta):
    # potentially have a constraint on the assortment size
    C = num_prods
    if 'max_assortment_size' in meta.keys():
        C = meta['max_assortment_size']

    price_list = ucm['p'][1:]
    start_time = time.time()
    price_sorted_products = (np.argsort(price_list) + 1)[::-1]
    HSet_idx = [tuple(np.where(xr)[0]) for xr in ucm['Hsets']]
    W_set = {tuple(sorted(list(np.where(key)[0]))): ucm['W'][key] for key in ucm['W'].keys()}
    maxRev, maxSet = 0, []
    maxIdx = -1
    for i in range(1, min(C + 1, len(price_sorted_products) + 1)):
        rev_ro_set = ucm_calc_revenue(price_sorted_products[:i], ucm['p'], ucm, num_prods, HSet_idx=HSet_idx,
                                      W_set=W_set)
        if rev_ro_set > maxRev:
            maxRev, maxSet, maxIdx = rev_ro_set, list(price_sorted_products[:i]), i + 1
    timeTaken = time.time() - start_time
    if meta.get('print_results', False) is True:
        logger.info(str((meta['algo'], 'revenue ordered rev:', maxRev, 'set:', maxSet, ' time taken:', timeTaken)))
    return maxRev, maxSet, timeTaken


def ucm_revenue_ordered(num_prods, C_old, ucm, meta):
    # potentially have a constraint on the assortment size
    C = num_prods
    if 'max_assortment_size' in meta.keys():
        C = meta['max_assortment_size']

    price_list = ucm['p'][1:]
    start_time = time.time()
    price_sorted_products = (np.argsort(price_list) + 1)[::-1]
    maxRev, maxSet = 0, []
    maxIdx = -1

    # setup for revenue calculation
    HSet_idx = [tuple(np.where(xr)[0]) for xr in ucm['Hsets']]
    W_set = {tuple(sorted(list(np.where(key)[0]))): ucm['W'][key] for key in ucm['W'].keys()}
    v2 = lambda i, j: ucm['v'][i] * ucm['v'][j] * np.exp(W_set[tuple(sorted([i - 1, j - 1]))]) if (
            tuple(sorted([i - 1, j - 1])) in HSet_idx) else ucm['v'][i] * ucm['v'][j]
    pos0 = np.zeros(len(ucm['v']) - 1)
    if (tuple(pos0) in ucm['Hsets']):
        v00 = ucm['v'][0] * ucm['v'][0] * np.exp(ucm['W'][tuple(pos0)])
    else:
        v00 = ucm['v'][0] * ucm['v'][0]

    num1, num2 = 0, 0
    den1, den2 = ucm['v'][0], v00

    for i in range(1, min(C + 1, len(price_sorted_products) + 1)):
        # update numerators
        st = time.time()
        curr_prod = price_sorted_products[i - 1]
        num1 += ucm['p'][curr_prod] * ucm['v'][curr_prod]
        den1 += ucm['v'][curr_prod]

        num2 += np.sum([(ucm['p'][price_sorted_products[xi]] + ucm['p'][curr_prod]) * (
            v2(price_sorted_products[xi], curr_prod)) for xi in range(i - 1)])

        den2 += np.sum([v2(price_sorted_products[xi], curr_prod) for xi in range(i - 1)])
        rev_ro_set = (ucm['probs_select_size'][1] * (num1 / den1)) + (ucm['probs_select_size'][2] * (num2 / den2))

        # print(f"revenue:{rev_ro_set} for set length {i} in {time.time() - st} secs...")
        # rev_ro_set_old = ucm_calc_revenue(price_sorted_products[:i], ucm['p'], ucm, num_prods, HSet_idx=HSet_idx,
        #                                   W_set=W_set)
        # print(f"revold:{rev_ro_set_old}, revnew: {rev_ro_set} for set length {i}")
        if rev_ro_set > maxRev:
            maxRev, maxSet, maxIdx = rev_ro_set, list(price_sorted_products[:i]), i + 1
    timeTaken = time.time() - start_time
    # if meta.get('print_results', False) is True:
    #     logger.info(str((meta['algo'], 'revenue ordered rev:', maxRev, 'set:', maxSet, ' time taken:', timeTaken)))
    return maxRev, maxSet, timeTaken


# ====================UCM Mixed Integer program ===================

def ucm_mixed_ip(num_prods, C, ucm, meta=None):
    start_time = time.time()
    opt_model = cpx.Model(name="UCM Model")

    # Declare Variables for Model
    p1_vars = \
        {i: opt_model.continuous_var(lb=0, ub=1,
                                     name=f"p_{i}") for i in range(num_prods + 1)}

    p2_vars = \
        {(i, j): opt_model.continuous_var(lb=0, ub=1,
                                          name=f"p_{i}_{j}") for i in range(1, num_prods + 1) for j in
         range(i + 1, num_prods + 1)}
    p2_vars.update({(0, 0): opt_model.continuous_var(lb=0, ub=1, name=f"p_{0}_{0}")})

    x_vars = \
        {(i, j): opt_model.binary_var(name=f"x_{i}_{j}") for i in range(1, num_prods + 1) for j in
         range(i, num_prods + 1)}

    # Add Constraints for Model
    constraintsA = {i: opt_model.add_constraint(ct=p1_vars[i] <= x_vars[i, i],
                                                ctname=f"constraint_p1_{i}<x{i}{i}")
                    for i in range(1, num_prods + 1)}

    constraintsA.update({(i, j): opt_model.add_constraint(
        ct=p2_vars[i, j] <= x_vars[i, j], ctname=f"constraint_p2_{i}{j}<x{i}{j}") for i in range(1, num_prods + 1) for j
        in
        range(i + 1, num_prods + 1)})

    constraintsB = {i: opt_model.add_constraint(ct=p1_vars[i] <= mixed_ip_get_v_var(ucm, i) * p1_vars[0],
                                                ctname=f"constraint_p1_{i}<=V{i}p1_{0}") for i in
                    range(1, num_prods + 1)}
    constraintsB.update({(i, j): opt_model.add_constraint(
        ct=p2_vars[i, j] <= mixed_ip_get_v_var(ucm, i, j) * p2_vars[0, 0],
        ctname=f"constraint_p2_{i}{j}<=V{i}{j}p2_{0}{0}") for
        i in range(1, num_prods + 1) for j in range(i + 1, num_prods + 1)})

    constraintsC = {i: opt_model.add_constraint(
        ct=p1_vars[i] + (mixed_ip_get_v_var(ucm, i) * (1 - x_vars[i, i])) >= mixed_ip_get_v_var(ucm, i) * p1_vars[0],
        ctname=f"constraint_p1_{i}+V{i}(1-x{i}{i})>=V{i}p1_{0}") for i in range(1, num_prods + 1)}
    constraintsC.update({(i, j): opt_model.add_constraint(
        ct=p2_vars[i, j] + (mixed_ip_get_v_var(ucm, i, j) * (1 - x_vars[i, j])) >= mixed_ip_get_v_var(ucm, i, j) *
           p2_vars[0, 0],
        ctname=f"constraint_p2_{i}{j}+V{i}{j}(1-x{i}{j})>=V{i}{j}p1_{0}{0}") for i in range(1, num_prods + 1) for j in
        range(i + 1, num_prods + 1)})

    constraintsD = {
        (i, j): opt_model.add_constraint(ct=x_vars[i, j] <= x_vars[i, i], ctname=f"constraint_x{i}{j}<=x{i}{i}") for i
        in range(1, num_prods + 1) for j in range(i + 1, num_prods + 1)}
    constraintsE = {
        (i, j): opt_model.add_constraint(ct=x_vars[i, j] <= x_vars[j, j], ctname=f"constraint_x{i}{j}<=x{j}{j}") for i
        in range(1, num_prods + 1) for j in range(i + 1, num_prods + 1)}
    constraintsF = {(i, j): opt_model.add_constraint(ct=x_vars[i, j] >= x_vars[i, i] + x_vars[j, j] - 1,
                                                     ctname=f"constraint_x{i}{j}>=x{i}{i}+x{j}{j}-1") for i in
                    range(1, num_prods + 1) for j in range(i + 1, num_prods + 1)}

    constraintsG = {1: opt_model.add_constraint(opt_model.sum(p1_vars[i] for i in range(num_prods + 1)) == 1,
                                                ctname=f"constraint_SUMpi=1"),
                    2: opt_model.add_constraint(p2_vars[0, 0] + opt_model.sum(
                        p2_vars[i, j] for i in range(1, num_prods + 1) for j in range(i + 1, num_prods + 1)) == 1,
                                                ctname=f"constraint_SUMpij=1")}

    # Add Objective Function
    objective = mixed_ip_get_z_val(ucm, 1) * opt_model.sum(
        mixed_ip_get_r_val(ucm, i) * p1_vars[i] for i in range(1, num_prods + 1)) + mixed_ip_get_z_val(ucm,
                                                                                                       2) * opt_model.sum(
        mixed_ip_get_r_val(ucm, i, j) * p2_vars[i, j] for i in range(1, num_prods + 1) for j in
        range(i + 1, num_prods + 1))

    # Solve given model
    opt_model.maximize(objective)
    # TODO: Fixing Url and API Key here, Need to get it from meta
    # meta['url'] = "https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/"
    # meta['key'] = "api_4e360631-3fd5-449b-b7a2-3449aca36a3b"
    opt_model.solve()
    time_taken = opt_model.solution._solve_details._time
    # get Optimal Set and Optimal Revenue
    x_solution = {(i, j): x_vars[i, j].solution_value for i in range(1, num_prods + 1) for j in range(i, num_prods + 1)}
    maxRev = opt_model.objective_value
    maxSet = [x_key[0] for x_key in x_solution.keys() if x_solution[x_key] == 1 and x_key[0] == x_key[1]]

    return maxRev, maxSet, time_taken


def mixed_ip_get_v_var(ucm, i, j=None):
    if not j:
        return ucm['v'][i] / ucm['v'][0]

    pos0 = np.zeros(len(ucm['v']) - 1)
    if (tuple(pos0) in ucm['Hsets']):
        v00 = ucm['v'][0] * ucm['v'][0] * np.exp(ucm['W'][tuple(pos0)])
    else:
        v00 = ucm['v'][0] * ucm['v'][0]
    pos = np.zeros(len(ucm['v']) - 1)
    pos[j - 1] = 1
    pos[i - 1] = 1
    if (tuple(pos) in ucm['Hsets']):
        return ucm['v'][i] * ucm['v'][j] * np.exp(ucm['W'][tuple(pos)]) / v00
    else:
        return ucm['v'][i] * ucm['v'][j] / v00


def mixed_ip_get_z_val(ucm, i):
    return ucm['probs_select_size'][i]


def mixed_ip_get_r_val(ucm, i, j=None):
    if j is None:
        return ucm['p'][i]
    return ucm['p'][i] + ucm['p'][j]


# ====================UCM Brute Search in all possible assortments===================

def ucm_brute_force_search(num_prods, C, ucm, meta=None, K=None):
    # best assortment for ground truth model: revenue
    start_time = time.time()
    rev_best = -1
    ast_best = []
    for size in range(1, C + 1):
        for ast in combinations(range(1, num_prods + 1), size):
            rev_temp = ucm_calc_revenue(ast, ucm['p'], ucm, num_prods)
            if rev_temp > rev_best:
                rev_best, ast_best = rev_temp, ast
    time_taken = time.time() - start_time
    return rev_best, ast_best, time_taken


# ====================UCM ADXOPT1 with products===================

def ucm_adxopt1_products(num_prods, C, ucm, meta=None):
    p = ucm['p']
    st = time.time()
    # initialize
    b = min(C, num_prods - C + 1)  # parameter of adxopt, see Thm 3, Jagabathula
    items = range(1, num_prods + 1)
    removals = np.zeros(num_prods + 1)
    if meta is not None:
        if meta.get('startadx', None) is not None:
            set_prev = sorted(meta['startadx'])
        else:
            set_prev = []
    else:
        set_prev = []
    rev_prev = ucm_calc_revenue(set_prev, p, ucm, num_prods)

    rev_cal_counter = 0
    while True:
        items_left = [x for x in items if x not in set_prev]
        # Additions
        set_addition = []
        rev_addition = 0
        if len(set_prev) < C:
            for j in items_left:
                if removals[j] < b:
                    candidate_rev = ucm_calc_revenue(sorted(set_prev + [j]), p, ucm, num_prods)
                    rev_cal_counter += 1
                    if candidate_rev > rev_addition:
                        rev_addition = candidate_rev
                        set_addition = sorted(set_prev + [j])

        # Deletions
        set_deletion = []
        rev_deletion = 0
        if len(set_prev) > 0:
            for idx in range(len(set_prev)):
                candidate_rev = ucm_calc_revenue(sorted(set_prev[:idx] + set_prev[idx + 1:]), p, ucm, num_prods)
                rev_cal_counter += 1
                if candidate_rev > rev_deletion:
                    rev_deletion = candidate_rev
                    set_deletion = sorted(set_prev[:idx] + set_prev[idx + 1:])

        # Substitutions
        set_substitution = []
        rev_substitution = 0
        if len(set_prev) > 0:
            for j in items_left:
                if removals[j] < b:
                    for idx in range(len(set_prev)):
                        candidate_rev = ucm_calc_revenue(sorted(set_prev[:idx] + [j] + set_prev[idx + 1:]), p, ucm,
                                                         num_prods)
                        rev_cal_counter += 1
                        if candidate_rev > rev_substitution:
                            rev_substitution = candidate_rev
                            set_substitution = sorted(set_prev[:idx] + [j] + set_prev[idx + 1:])

        idx_rev_current = np.argmax(np.asarray([rev_addition, rev_deletion, rev_substitution]))
        if idx_rev_current == 0:
            rev_current = rev_addition
            set_current = set_addition
        elif idx_rev_current == 1:
            rev_current = rev_deletion
            set_current = set_deletion
        else:
            rev_current = rev_substitution
            set_current = set_substitution

        if rev_current <= rev_prev or np.min(removals) >= b:
            rev_current = rev_prev
            set_current = set_prev
            break
        else:
            for j in set(set_prev).difference(set(set_current)):
                removals[j] += 1

            rev_prev = rev_current
            set_prev = set_current

    timeTaken = time.time() - st

    # print(set_current,p,ucm,num_prods)
    rev_adx = ucm_calc_revenue(set_current, p, ucm, num_prods)
    logger.info("Number of times ucm_calc_revenue is called: %d" % rev_cal_counter)
    logger.info('rev adx: %.3f' % rev_adx)
    logger.info("Products in the adxopt assortment are %s" % str(set_current))
    logger.info('Time taken for running adxopt is %.3f secs...' % timeTaken)

    return rev_adx, set_current, timeTaken


# ====================RCM ADXOPT2 with subsets===================

def ucm_adxopt2_sets(num_prods, C, ucm, meta=None, two_sets=False, b=None, allow_exchange=True):
    p = ucm['p']
    st = time.time()
    # initialize
    if b is None:
        b = min(C, num_prods - C + 1)  # This is the parameter for MNL, figure out the correct parameter for RCM
    items = range(1, num_prods + 1)
    # all_sets = combinations(items, 1)
    # if two_sets:
    #     all_sets = chain(all_sets, combinations(items, 2))

    # removals = {}
    # for subset in all_sets:
    #    removals[subset] = 0
    removals = np.zeros(num_prods + 1)
    if meta is not None:
        if meta.get('startadx', None) is not None:
            set_prev = sorted(meta['startadx'])
        else:
            set_prev = []
    else:
        set_prev = []
    rev_prev = ucm_calc_revenue(set_prev, p, ucm, num_prods)

    rev_cal_counter = 0
    while True:
        items_left = [x for x in items if (x not in set_prev) & (removals[x] < b)]
        sets_left = combinations(items_left, 1)
        if two_sets:
            sets_left = chain(sets_left, combinations(items_left, 2))
        sets_left = list(sets_left)
        # Additions
        set_addition = []
        rev_addition = 0
        if len(set_prev) < C:
            for j in sets_left:  # items_left:
                # if np.all([removals[k] <b for k in j]):
                if len(set_prev) + len(list(j)) <= C:
                    candidate_rev = ucm_calc_revenue(sorted(set_prev + list(j)), p, ucm, num_prods)
                    # print 'Considered set' , j, 'has revenue ', candidate_rev
                    rev_cal_counter += 1
                    if candidate_rev > rev_addition:
                        rev_addition = candidate_rev
                        set_addition = sorted(set_prev + list(j))
                        # print 'set_addition is', set_addition

        # Deletions
        set_deletion = []
        rev_deletion = 0
        if len(set_prev) > 0:
            #             for idx in range(len(set_prev)):
            #                 candidate_rev = ucm_calc_revenue(sorted(set_prev[:idx]+set_prev[idx+1:]),p,ucm,num_prods)
            #                 rev_cal_counter +=1
            #                 if candidate_rev > rev_deletion:
            #                     rev_deletion = candidate_rev
            #                     set_deletion = sorted(set_prev[:idx]+set_prev[idx+1:])
            avail_subsets = combinations(set_prev, 1)
            if two_sets:
                avail_subsets = chain(avail_subsets, combinations(set_prev, 2))
            avail_subsets = list(avail_subsets)
            for subset in avail_subsets:
                # print sorted(list(np.setdiff1d( set_prev , subset )))
                candidate_rev = ucm_calc_revenue(sorted(list(np.setdiff1d(set_prev, subset))), p, ucm, num_prods)
                rev_cal_counter += 1
                if candidate_rev > rev_deletion:
                    rev_deletion = candidate_rev
                    set_deletion = sorted(list(np.setdiff1d(set_prev, subset)))

        # Substitutions
        set_substitution = []
        rev_substitution = 0
        if allow_exchange:
            if len(set_prev) > 0:
                for j in sets_left:
                    # if np.all([removals[k] <b for k in j]):#removals[j] <b:
                    # for idx in range(len(set_prev)):
                    #    candidate_rev = ucm_calc_revenue(sorted(set_prev[:idx]+[j]+set_prev[idx+1:]),p,ucm,num_prods)
                    for subset in avail_subsets:
                        new_ast = sorted(list(np.setdiff1d(set_prev, subset)) + list(j))
                        if len(new_ast) <= C:
                            candidate_rev = ucm_calc_revenue(new_ast, p, ucm, num_prods)
                            rev_cal_counter += 1
                            if candidate_rev > rev_substitution:
                                rev_substitution = candidate_rev
                                set_substitution = new_ast  # sorted(set_prev[:idx]+[j]+set_prev[idx+1:])

        idx_rev_current = np.argmax(np.asarray([rev_addition, rev_deletion, rev_substitution]))
        if idx_rev_current == 0:
            rev_current = rev_addition
            set_current = set_addition

        elif idx_rev_current == 1:
            rev_current = rev_deletion
            set_current = set_deletion
        else:
            rev_current = rev_substitution
            set_current = set_substitution

        if rev_current <= rev_prev or np.min(removals) >= b:
            rev_current = rev_prev
            set_current = set_prev
            break
        else:
            for j in set(set_prev).difference(set(set_current)):
                # for k in j:
                removals[j] += 1

            rev_prev = rev_current
            set_prev = set_current
    timeTaken = time.time() - st

    rev_adx = ucm_calc_revenue(set_current, p, ucm, num_prods)
    if meta is not None:
        if meta.get('print_results', False) is not False:
            logger.info("Number of times ucm_calc_revenue is called: %d" % rev_cal_counter)
            logger.info('rev adx:%.3f' % rev_adx)
            logger.info("Products in the adxopt assortment are %s" % str(set_current))
            logger.info('Time taken for running adxopt is %.3f secs..' % timeTaken)

    return rev_adx, set_current, timeTaken


# ==================== Auxiliary Functions ===================
def ucm_get_assortment_probs(given_set, ucm, prod):
    set2idx = {}
    idx2set = {}
    idx2set[0] = 0
    set2idx[0] = 0
    for idx, x in enumerate(given_set):
        set2idx[x] = idx + 1
        idx2set[idx + 1] = x

    # Obtain probabilities
    probs = collections.OrderedDict({x: {} for x in range(1, len(given_set) + 1)})
    probs_select_size_new = {}
    probs_select_size_new[0] = ucm['probs_select_size'][0]
    temp_selection_probsum = ucm['probs_select_size'][0]
    for size in range(1, min(ucm['max_purchase_size'], len(given_set)) + 1):
        index_subsets = [tuple([0] * size)] + list(combinations(range(1, len(given_set) + 1), size))
        # print(index_subsets)
        for ast in index_subsets:
            set_char_vector = set_char_from_ast([idx2set[x] for x in ast], prod)
            # print(ast, set_char_vector)
            probs[size][set_char_vector] = 1
            for x in ast:
                probs[size][set_char_vector] *= ucm['v'][idx2set[x]]
            if (set_char_vector in ucm['Hsets']) & (size > 1):
                probs[size][set_char_vector] *= np.exp(ucm['W'][set_char_vector])

        normalization_temp = sum([x for x in probs[size].values()])
        # print(normalization_temp)
        for ast in index_subsets:
            set_char_vector = set_char_from_ast([idx2set[x] for x in ast], prod)
            probs[size][set_char_vector] /= normalization_temp

    #     temp_selection_probsum += ucm['probs_select_size'][size]
    #     probs_select_size_new[size] = ucm['probs_select_size'][size]
    #
    # for size in probs_select_size_new:
    #     probs_select_size_new[size] /= temp_selection_probsum

    return probs


def ucm_calc_revenue_old(given_set, p, ucm, prod):
    probs = ucm_get_assortment_probs(given_set, ucm, prod)
    probs_select_size_new = ucm['probs_select_size']
    rev = 0

    for size in range(1, len(given_set) + 1):
        for set_char_vector in probs[size]:
            subset_price = 0
            for i in range(len(set_char_vector)):
                if set_char_vector[i] == 1:
                    subset_price += p[i + 1]
            prob_subset = probs[size][set_char_vector] * probs_select_size_new[size]
            # print(size, subset_price, set_char_vector, prob_subset)
            # print(set_char_vector,'subset_price',subset_price,'prob_subset',prob_subset)
            rev = rev + subset_price * prob_subset
    return rev


def ucm_calc_revenue(given_set, p, ucm, prod, HSet_idx=None, W_set=None):
    start_time = time.time()
    v2 = lambda i, j: ucm['v'][i] * ucm['v'][j] * np.exp(W_set[tuple(sorted([i - 1, j - 1]))]) if (
            tuple(sorted([i - 1, j - 1])) in ([] if HSet_idx is None else HSet_idx)) else ucm['v'][i] * ucm['v'][j]

    pos0 = np.zeros(len(ucm['v']) - 1)
    if (tuple(pos0) in ucm['Hsets']):
        v00 = ucm['v'][0] * ucm['v'][0] * np.exp(ucm['W'][tuple(pos0)])
    else:
        v00 = ucm['v'][0] * ucm['v'][0]

    num1, num2 = 0, 0
    den1, den2 = ucm['v'][0], v00

    num1 += np.sum([ucm['p'][xr] * ucm['v'][xr] for xr in given_set])
    den1 += np.sum([ucm['v'][xr] for xr in given_set])

    num2 += np.sum([(ucm['p'][given_set[xi]] + ucm['p'][given_set[xj]]) * (v2(given_set[xi], given_set[xj]))
                    for xi in range(len(given_set)) for xj in range(xi + 1, len(given_set))])

    den2 += np.sum(
        [v2(given_set[xi], given_set[xj]) for xi in range(len(given_set)) for xj in range(xi + 1, len(given_set))])

    revenue = (ucm['probs_select_size'][1] * (num1 / den1)) + (ucm['probs_select_size'][2] * (num2 / den2))
    # print(f"revenue:{revenue} for set length {len(given_set)} in {time.time() - start_time} secs...")
    return revenue

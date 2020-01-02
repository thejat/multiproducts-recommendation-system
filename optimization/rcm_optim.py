import numpy as np
import time
from datetime import datetime
import os
from itertools import combinations
from itertools import chain
# from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
from synthetic_models.utils import set_char_from_ast, ast_from_set_char
import cplex
import docplex.mp.model as cpx
from threading import Thread, Lock
import pickle
import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
from synthetic_models.tcm_abstract_model import model as tcm_model
from pyomo.environ import *
import functools
import multiprocessing as mp

opt = SolverFactory('bonmin')

logger = logging.getLogger(__name__)

'''
Algorithms for RCM Optimization
A: Binary Search With:
B: MIP Formulation
C: adxOpt1_products
D: adxOpt2_sets
E: Brute Search
'''


def init_optim_algorithms():
    global optim_methods_src
    optim_methods_src = {
        'binary-search-improved': rcm_binary_search_v2,
        'binary-search': rcm_binary_search,
        'mixed-ip': rcm_mixed_ip,
        'adxopt1': rcm_adxopt1_products,
        'adxopt2': rcm_adxopt2_sets,
        'revenue-ordered': rcm_revenue_ordered,
        'mnl-revenue-ordered': mnl_revenue_ordered,
        'brute-force': rcm_brute_force_search,
        'tcm_bonmin_mnlip': tcm_bonmin_mnlip
    }


def run_rcm_optimization(algorithm, num_prods, C, rcm_model, meta):
    init_optim_algorithms()
    global optim_methods_src
    optim_function = optim_methods_src[algorithm]
    maxRev, maxSet, timeTaken, solve_log = optim_function(num_prods, C, rcm_model, meta)
    if (type(timeTaken) == dict):
        time_log = timeTaken
        timeTaken = time_log['total_time_taken']
    else:
        time_log = {'total_time_taken': timeTaken}
    if 'comparison_function' in meta.keys():
        logger.info(
            f"Binary Search: {meta['comparison_function']},MaxRev: {maxRev},MaxSet: {maxSet}, TimeTaken: "
            f"{str(timeTaken)}")
    else:
        logger.info(f"Algorithm: {algorithm},MaxRev: {maxRev},MaxSet: {maxSet}, TimeTaken:{str(timeTaken)}")
    return {'max_revenue': maxRev, 'max_set': maxSet, 'time_taken': timeTaken, 'time_log': time_log,
            'solve_log': str(solve_log)}


'''
Algorithm 1: Binary Search Algorithm
Comparision Step Methods:
  1- NN Exact Comparision
  2- NN Approximate Comparision
  3- QIP Absolute Comparision
  4- QIP heuristics Comparision
  5- IP Absolute Comparision
'''


def init_comparision_methods():
    global binSearch_comparision_src
    binSearch_comparision_src = {
        'nn-exact': binSearchCompare_nn,
        'nn-approx': binSearchCompare_nn,
        'qip-exact': binSearchCompare_qip_exact,
        'qip-approx-spc': binSearchCompare_qip_approx_spc,
        'qip-approx': binSearchCompare_qip_approx_multithread,
        'ip-exact': binSearchCompare_ip_exact
    }


# ====================MNL Revenue Ordered Assortments ====================================
def mnl_revenue_ordered(num_prods, C, rcm, meta):
    improved_lower_bound = 0
    price_list = rcm['p'][1:]
    start_time = time.time()
    price_sorted_products = (np.argsort(price_list) + 1)[::-1]
    maxRev, maxSet = 0, []
    maxIdx = -1
    numerator_i, denominator_i = 0., rcm['v'][0]
    for i in range(1, len(price_sorted_products) + 1):
        # rev_ro_set = mnl_calc_revenue(price_sorted_products[:i], rcm['p'], rcm['v'])
        numerator_i += rcm['p'][price_sorted_products[i - 1]] * rcm['v'][price_sorted_products[i - 1]]
        denominator_i += rcm['v'][price_sorted_products[i - 1]]
        rev_ro_set = numerator_i / denominator_i
        if rev_ro_set > maxRev:
            maxRev, maxSet, maxIdx = rev_ro_set, list(price_sorted_products[:i]), i + 1
    timeTaken = time.time() - start_time
    if meta.get('print_results', False) is True:
        logger.info(str((meta['algo'], 'revenue ordered rev:', maxRev, 'set:', maxSet, ' time taken:', timeTaken)))
    solve_log = {'max_idx': maxIdx}
    return maxRev, maxSet, timeTaken, solve_log


#
# def mnl_calc_revenue(product_list, p, v):
#     num = np.sum([p[prod] * v[prod] for prod in product_list])
#     den = np.sum([v[0]] + [v[prod] for prod in product_list])
#     return num / den


# ====================RCM Revenue Ordered Assortments ====================================
def rcm_revenue_ordered(num_prods, C, rcm, meta):
    # potentially have a constraint on the assortment size
    C = num_prods
    if 'max_assortment_size' in meta.keys():
        C = meta['max_assortment_size']

    price_list = rcm['p'][1:]
    start_time = time.time()
    price_sorted_products = (np.argsort(price_list) + 1)[::-1]
    maxRev, maxSet = 0, []
    maxIdx = -1
    den0, den1, den2 = rcm['v'][0], 0, 0
    num1, num2 = 0, 0
    for i in range(1, min(C + 1, len(price_sorted_products) + 1)):
        # rev_ro_set = rcm_calc_revenue(price_sorted_products[:i], rcm['p'], rcm, num_prods)
        curr_prod = price_sorted_products[i - 1]
        num1 += rcm['p'][curr_prod] * rcm['v'][curr_prod]
        num2 += sum(
            [(rcm['p'][price_sorted_products[xj]] + rcm['p'][curr_prod]) * (
                rcm['v2'][tuple([price_sorted_products[xj], curr_prod])]) for xj in range(i - 1)])
        den1 += rcm['v'][curr_prod]
        den2 += sum([(rcm['v2'][tuple([price_sorted_products[xj], curr_prod])]) for xj in range(i - 1)])
        # print(rev_ro_set, (num1 + num2) / (den0 + den1 + den2))
        #todo: Ask Theja/Deeksha about this change
        rev_ro_set = rcm_calc_revenue(price_sorted_products[:i], rcm['p'], rcm, num_prods)
        if rev_ro_set > maxRev:
            maxRev, maxSet, maxIdx = rev_ro_set, list(price_sorted_products[:i]), i + 1
    timeTaken = time.time() - start_time
    if meta.get('print_results', False) is True:
        logger.info(str((meta['algo'], 'revenue ordered rev:', maxRev, 'set:', maxSet, ' time taken:', timeTaken)))
    solve_log = {'max_idx': maxIdx}
    return maxRev, maxSet, timeTaken, solve_log


# ====================RCM Improved Binary Search(with lemmas) Algorithm===================

def rcm_binary_search_v2(num_prods, C, rcm, meta):
    '''
            binary search using different comparison functions: get_nn_set_rcm or capAst_QPcompare_rcm
        '''
    # Not using K value at the moment. Binary search is irrelevant
    init_comparision_methods()
    global binSearch_comparision_src
    comparison_function = binSearch_comparision_src[meta['comparison_function']]
    p = rcm['p']

    if meta.get('eps', None) is None:
        meta['eps'] = 1e-3
    if 'nn' in meta['comparison_function']:
        logger.info("Support Not Available for NN Function in imporved Binary Search...")
        raise Exception
    clusters_allowed = False
    if 'clusters_allowed' in meta.keys():
        if (meta['clusters_allowed']) & ('max_problem_size' in meta.keys()):
            clusters_allowed = True
    if clusters_allowed:
        meta['num_clusters'], meta['cluster_ids'] = binSearchClusterProducts(num_prods, C, rcm, meta)

    st = time.time()
    solve_time = 0.
    solve_log = {}
    time_log = {}
    count = 0
    maxSet = None
    meta['is_improved_qubo'] = 1
    ##Heuristic to get optimal Value for L based on Lemma 3"
    start_time = time.time()
    L = binSearchImproved_global_lower_bound(num_prods, C, rcm, meta)
    logger.info(f"time taken to get global lower bound {(time.time() - start_time) * 1e6} microsecs")
    time_log['global_bound'] = (time.time() - start_time) * 1e6
    logger.debug(f"Improved Global Lower Bound For Binary Search : {L}")
    solve_log['global_lower_bound'] = L
    U = 2 * max(p)  # U is the upper bound on the objective
    iter_count = 0
    logger.debug(f"Starting Binary Search with comparision function:{meta['comparison_function']} U: {U},L:{L}....")
    while (U - L) > meta['eps']:
        logger.info(f"\niteration: {iter_count}")
        count += 1
        K = (U + L) / 2

        start_time = time.time()
        removed_products = binSearchImproved_removed_products(num_prods, C, rcm, meta, L)
        logger.info(f"time taken to remove products {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'I{count}___remove_products'] = (time.time() - start_time) * 1e6

        start_time = time.time()
        selected_products = binSearchImproved_selected_products(num_prods, C, rcm, meta, K)
        logger.info(f"time taken to select products {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'I{count}___select_products'] = (time.time() - start_time) * 1e6

        logger.debug(f"Iteration: {iter_count}; U,L:"
                     f" {U},{L}; #products removed:{len(removed_products)},selected: {len(selected_products)}")

        start_time = time.time()
        meta['selected_products'] = selected_products
        meta['removed_products'] = removed_products
        if (len(selected_products + removed_products) >= num_prods):
            maxSet = selected_products
            break
        logger.info(f"time taken to setup meta {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'I{count}___meta_setup'] = (time.time() - start_time) * 1e6

        start_time = time.time()
        maxPseudoRev, maxSet, queryTimeLog = comparison_function(num_prods, C, rcm, meta, K)
        logger.info(f"time taken in comparision step {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'I{count}___comparision'] = (time.time() - start_time) * 1e6
        for key in queryTimeLog.keys():
            time_log[f'I{count}___compstep_{key}'] = queryTimeLog[key]

        start_time = time.time()
        iter_count += 1
        solve_time += time_log[f'I{count}___comparision']
        # logger.info('pseudorev/vo',maxPseudoRev/rcm['v'][0],'K:',K,' U:',U, ' L:',L)
        # Add Selected products in mix
        maxSet = list(set(maxSet + meta['selected_products']))
        maxRev = rcm_calc_revenue(maxSet, p, rcm, num_prods)
        logger.info(f"time taken aftermath comparision step(calc rev) {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'I{count}___aftercomp_revenue'] = (time.time() - start_time) * 1e6

        start_time = time.time()
        logger.debug({f"K:{K}, MaxRev: {maxRev}, Time Taken: {queryTimeLog}"})
        logger.debug({f"MaxSet: {maxSet}"})
        solve_log[f'iter_{iter_count}'] = {
            'U': U,
            'L': L,
            'removed_product_count': len(removed_products),
            'selected_product_count': len(selected_products),
            'optim_product_count': num_prods - (len(removed_products) + len(selected_products)),
            'comp_step_time': queryTimeLog
        }
        logger.info(f"time taken aftermath comparision step(logging) {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'I{count}___aftercomp_logging'] = (time.time() - start_time) * 1e6

        if maxRev >= K:
            L = K
        else:
            U = K
    logger.info(f" Binary Search Improved Loop Done..")

    start_time = time.time()
    maxRev = rcm_calc_revenue(maxSet, p, rcm, num_prods)
    logger.info(f"time taken afterloop (revenue calc) {(time.time() - start_time) * 1e6} microsecs")
    time_log[f'afterloop_revenue'] = (time.time() - start_time) * 1e6
    timeTaken = time.time() - st
    time_log['total_time_taken'] = timeTaken
    solve_log['solveTime'] = solve_time
    solve_log['setupTime'] = timeTaken - solve_time
    if meta.get('print_results', False) is True:
        logger.info(
            f"Total Time Taken: {timeTaken} secs, Solve Time: {solve_time} secs, Setup Time: {timeTaken - solve_time} secs")
    #     logger.info(str((meta['algo'], 'binary search rev:', maxRev, 'set:', maxSet, ' time taken:', timeTaken,
    #                      ' num iters:', count)))
    return maxRev, maxSet, time_log, solve_log


def binSearchImproved_global_lower_bound(num_prods, C, rcm, meta):
    improved_lower_bound = 0
    price_list = rcm['p'][1:]
    price_sorted_products = (np.argsort(price_list) + 1)[::-1]
    rev_last_ro_set = rcm_calc_revenue(price_sorted_products[:1], rcm['p'], rcm, num_prods)
    for i in range(2, len(price_sorted_products)):
        rev_next_ro_set = rcm_calc_revenue(price_sorted_products[:i], rcm['p'], rcm, num_prods)
        if rev_next_ro_set + 1e-5 < rev_last_ro_set:
            improved_lower_bound = rcm['p'][price_sorted_products[i]]
            break
        rev_last_ro_set = rev_next_ro_set
    return improved_lower_bound


def binSearchImproved_removed_products(num_prods, C, rcm, meta, lower_limit):
    price_list = rcm['p'][1:]
    price_sorted_products_inc = (np.argsort(price_list) + 1)
    removed_products = []
    product_max_price = rcm['p'][price_sorted_products_inc[-1]]
    for i in range(len(price_sorted_products_inc) - 1):
        if ((rcm['p'][price_sorted_products_inc[i]] + product_max_price) < lower_limit):
            removed_products.append(price_sorted_products_inc[i])
        else:
            break
    return removed_products


def binSearchImproved_selected_products(num_prods, C, rcm, meta, upper_limit):
    price_list = rcm['p'][1:]
    price_sorted_products = (np.argsort(price_list) + 1)[::-1]
    selected_products = []
    for i in range(len(price_sorted_products)):
        if (rcm['p'][price_sorted_products[i]]) > upper_limit:
            selected_products.append(price_sorted_products[i])
        else:
            break
    return selected_products


def binSearchClusterProducts(num_prods, C, rcm, meta):
    V_mat = np.zeros((num_prods, num_prods))
    for i in range(num_prods):
        V_mat[i, i] = rcm['v'][i + 1]
        for j in range(i + 1, num_prods):
            V_mat[i, j] = rcm['v2'][tuple([i + 1, j + 1])]
            V_mat[j, i] = V_mat[i, j]

    max_cluster_variable_count = V_mat.shape[0]
    cluster_labels = np.zeros(max_cluster_variable_count)

    # Break Problem in Parts based on nearest block diagonalization
    num_clusters = 1
    while (max_cluster_variable_count > meta['max_problem_size']):
        num_clusters += 1
        spc = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
        spc.fit(V_mat)
        max_cluster_variable_count = pd.Series(spc.labels_).value_counts().iloc[0]
        cluster_labels = spc.labels_

    return num_clusters, cluster_labels


# ====================RCM Binary Search Algorithm===================

def rcm_binary_search(num_prods, C, rcm, meta):
    '''
        binary search using different comparison functions: get_nn_set_rcm or capAst_QPcompare_rcm
    '''
    # Not using K value at the moment. Binary search is irrelevant
    init_comparision_methods()
    global binSearch_comparision_src
    comparison_function = binSearch_comparision_src[meta['comparison_function']]
    p = rcm['p']
    # L = 0  # L is the lower bound on the objective
    # temp:
    solve_log = {}
    L = meta['eps']
    U = 2 * max(p)  # U is the upper bound on the objective
    solve_log['init_L'] = L
    solve_log['init_U'] = U
    if meta.get('eps', None) is None:
        meta['eps'] = 1e-3
    if 'nn' in meta['comparison_function']:
        # Try to find Index in Cache
        index_cache_filepath = meta['index_filepath']
        if os.path.exists(index_cache_filepath):
            logger.info(f"Picked Index From cache file {index_cache_filepath}..")
            index_dict = pickle.load(open(index_cache_filepath, 'rb'))
            comp_function = meta['comparison_function']
            if (comp_function == 'nn-exact'):
                index_type = 'exact_index'
            else:
                index_type = 'approx_index'
            meta['db'], meta['normConst'], meta['ptsTemp'] = index_dict[index_type]['db'], \
                                                             index_dict[index_type]['normConst'], \
                                                             index_dict[index_type]['ptsTemp']
        else:
            meta['db'], _, meta['normConst'], meta['ptsTemp'] = compare_nn_preprocess(num_prods, C, rcm['p'],
                                                                                      meta['comparison_function'])
    clusters_allowed = False
    if 'clusters_allowed' in meta.keys():
        if (meta['clusters_allowed']) & ('max_problem_size' in meta.keys()):
            clusters_allowed = True
    if clusters_allowed:
        meta['num_clusters'], meta['cluster_ids'] = binSearchClusterProducts(num_prods, C, rcm, meta)

    st = time.time()
    iter_count = 0
    solve_time = 0
    count = 0
    maxSet = None
    while (U - L) > meta['eps']:
        count += 1
        K = (U + L) / 2
        maxPseudoRev, maxSet, queryTimeLog = comparison_function(num_prods, C, rcm, meta, K)
        # logger.info('pseudorev/vo',maxPseudoRev/rcm['v'][0],'K:',K,' U:',U, ' L:',L)
        solve_time += queryTimeLog
        iter_count += 1
        solve_log[f'iter_{iter_count}'] = {
            'U': U,
            'L': L,
            'comp_step_time': queryTimeLog
        }
        if (maxPseudoRev / rcm['v'][0]) >= K:
            L = K
        else:
            U = K
    maxRev = rcm_calc_revenue(maxSet, p, rcm, num_prods)
    timeTaken = time.time() - st
    solve_log['solve_time'] = solve_time
    solve_log['setup_time'] = timeTaken - solve_time
    if meta.get('print_results', False) is True:
        logger.info(
            f"Total Time Taken: {timeTaken} secs, Solve Time: {solve_time} secs, Setup Time: {timeTaken - solve_time} secs")
        # print(meta['algo'], 'binary search rev:', maxRev, 'set:', maxSet, ' time taken:', timeTaken,
        #       ' num iters:',
        #       count)
    return maxRev, maxSet, timeTaken, solve_log


# --------------------BinSearch Compare Step: Quadratic Integer Programming -------------------
def binSearchCompare_qip_exact(num_prods, C, rcm, meta, K):
    '''
        QUBO integer programming comparison
        We will use the cplex 12.9 python api and also generate constraints using the column based technique
    '''

    p = cplex.Cplex()
    p.objective.set_sense(p.objective.sense.maximize)
    time_log = {}
    # potentially have a constraint on the assortment size
    max_assortment_size = num_prods
    if 'max_assortment_size' in meta.keys():
        max_assortment_size = meta['max_assortment_size']

    # apply improvements along with max_assortment_size
    start_time = time.time()
    if 'is_improved_qip' in meta.keys():
        selected_products = sorted(meta['selected_products'])  # assumed 1 indexing
        removed_products = sorted(meta['removed_products'])  # assumed 1 indexing
        rhs_values = [max_assortment_size] + [1] * len(selected_products) + [0] * len(removed_products)
        sense_values = ['L'] + ['E'] * (len(selected_products) + len(removed_products))
        p.linear_constraints.add(rhs=rhs_values, senses=sense_values)

        cols = []
        ks, kr = 1, len(selected_products) + 1
        for i in range(1, num_prods + 1):
            if i in selected_products:
                cols.append([[0, ks], [1.0, 1.0]])
                ks += 1
            elif i in removed_products:
                cols.append([[kr], [1.0]])
                kr += 1
            else:
                cols.append([[0], [1.0]])
        # print(kr)
        # print(1+len(selected_products)+len(removed_products))
        # print(cols)
        # print(rhs_values)
        # print(sense_values)
        assert (kr == 1 + len(selected_products) + len(
            removed_products)), 'Issue with number of rows and selected and removed products'
        logger.info(f"time taken in setting up selected/removed vars {(time.time() - start_time) * 1e6} microsecs")
        time_log['selected_removed_vars'] = (time.time() - start_time) * 1e6
    else:
        p.linear_constraints.add(rhs=[max_assortment_size], senses="L")
        cols = [[[0], [1.0]] for i in range(num_prods)]
        logger.info(f"time taken in setting up non-improved vars {(time.time() - start_time) * 1e6} microsecs")
        time_log['non_improved_vars'] = (time.time() - start_time) * 1e6
        '''
        Here each row of the cols list represents a two element list.
        the first element is a list of rows/constraint indices where the variable appears
        the second element is the coefficient of this variable in that row/constraint
        '''
    start_time = time.time()
    obj = np.multiply(np.array(rcm['v'][1:]), np.array(rcm['p'][1:])) - K * np.array(rcm['v'][1:])
    obj = obj.tolist()
    ub = [1 for i in range(num_prods)]
    types = ''.join(['I' for i in range(num_prods)])
    names = ["x_" + str(i + 1) for i in range(num_prods)]
    # print(obj)
    # print(ub)
    # print(types)
    # print(names)

    p.variables.add(obj=obj, ub=ub, columns=cols, types=types, names=names)
    logger.info(f"time taken in creating vars, objectives {(time.time() - start_time) * 1e6} microsecs")
    time_log['create_vars_obj'] = (time.time() - start_time) * 1e6

    start_time = time.time()
    qmat = []
    for idxi in range(num_prods):
        temprow = [[], []]
        for idxj in range(num_prods):
            if idxi == idxj:
                continue
            temprow[0].append(idxj)
            temprow[1].append((rcm['p'][idxi + 1] + rcm['p'][idxj + 1] - K) * rcm['v2'][(idxi + 1, idxj + 1)])
        qmat.append(temprow)
    p.objective.set_quadratic(qmat)
    logger.info(f"time taken in setting up constraints {(time.time() - start_time) * 1e6} microsecs")
    time_log['add_constraints'] = (time.time() - start_time) * 1e6

    # temporarily enabling the following 8 lines
    # p.write("qip.lp")
    # logger.info(qmat)
    # logger.info(types)
    # logger.info(p.variables.get_lower_bounds())
    # logger.info(p.variables.get_upper_bounds())
    # logger.info(p.variables.get_names())
    # logger.info(p.linear_constraints.get_num())
    # logger.info(p.variables.get_num())

    p.set_log_stream(None)
    p.set_error_stream(None)
    p.set_warning_stream(None)
    p.set_results_stream(None)
    start_time = time.time()
    st = time.time()
    p.solve()
    logger.info(f"time taken in solving QIP {(time.time() - start_time) * 1e6} microsecs")
    time_log['solve_qip'] = (time.time() - start_time) * 1e6

    # logger.info("\t\tQIP cplex status = ", p.solution.get_status(), ":", p.solution.status[p.solution.get_status()])
    # logger.info(p.solution.get_values())
    start_time = time.time()
    pseudoRev = p.solution.get_objective_value()
    timeTaken = time.time() - st
    # logger.info('\t\tTime taken for running the QIP is', timeTaken)

    revSet = []
    x = p.solution.get_values()
    for i in range(num_prods):
        if x[i] > 1e-3:
            revSet.append(int(i + 1))
    logger.info(f"time taken in postprocessing {(time.time() - start_time) * 1e6} microsecs")
    time_log['qip_postprocessing'] = (time.time() - start_time) * 1e6
    # logger.info("\t\tQIP pseudo rev:",pseudoRev)
    # logger.info("\t\tQIP set:", revSet)
    return pseudoRev, revSet, time_log


def binSearchCompare_qip_approx_spc(num_prods, C, rcm, meta, K):
    maxcut_heuristic_list = ['BURER2002', 'FESTA2002G', 'FESTA2002GPR', 'FESTA2002VNS', 'FESTA2002VNSPR',
                             'FESTA2002GVNS',
                             'FESTA2002GVNSPR', 'DUARTE2005', 'LAGUNA2009CE', 'LAGUNA2009HCE']

    input_filename = meta['QIPApprox_input']
    output_filename = meta['QIPApprox_output']
    heuristic_list = meta['heuristic_list']
    is_debug = meta['print_debug']
    MQLib_dir = meta['MQLib_dir']
    p_arr = rcm['p']
    v_arr = rcm['v']
    vij_arr = rcm['v2']
    is_improved_qubo = False

    if 'is_improved_qubo' in meta.keys():
        is_improved_qubo = True
        old2new_index, new2old_index = {}, {}
        selected_products = meta['selected_products']
        removed_products = meta['removed_products']
        not_allowed_products = selected_products + removed_products
        new_product_count = num_prods - len(not_allowed_products)
    else:
        selected_products = None
        removed_products = None
        not_allowed_products = None
        new_product_count = num_prods

    # Create Empty Q matrix with appropriate size
    num_variables = new_product_count
    Q_mat = np.zeros((num_variables, num_variables))

    if not is_improved_qubo:
        # Setup Input File for approx step
        for i in range(num_prods):
            for j in range(i, num_prods):
                Q_mat[i, j] = compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr, K)
    else:
        # SetupIndex transfer
        new_index = 0
        for i in range(num_prods):
            if i + 1 in not_allowed_products:
                continue
            else:
                old2new_index[i] = new_index
                new_index += 1

        for key, val in old2new_index.items():
            new2old_index[val] = key

        for i in range(num_prods):
            if (i + 1) in removed_products:
                continue
            elif (i + 1) in selected_products:
                for j in range(i, num_prods):
                    if (j + 1) not in not_allowed_products:
                        Q_mat[old2new_index[j], old2new_index[j]] += compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr,
                                                                                          K)
            else:  # i not in no optim products
                for j in range(i, num_prods):
                    if (j + 1) in removed_products:
                        continue
                    elif (j + 1) in selected_products:
                        Q_mat[old2new_index[i], old2new_index[i]] += compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr,
                                                                                          K)
                    else:  # J not in no optim products
                        Q_mat[old2new_index[i], old2new_index[j]] += compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr,
                                                                                          K)

    maxSetMap = {}
    maxRevMap = {}
    Q_mat_index = np.arange(num_variables)
    if is_improved_qubo:
        new_cluster_ids = np.zeros(len(old2new_index.keys()))
        for i in range(new_cluster_ids.shape[0]):
            new_cluster_ids[i] = meta['cluster_ids'][new2old_index[i]]
        Q_mat_label_map = {cluster_id: Q_mat_index[new_cluster_ids == cluster_id] for cluster_id in
                           range(meta['num_clusters'])}
    else:
        Q_mat_label_map = {cluster_id: Q_mat_index[meta['cluster_ids'] == cluster_id] for cluster_id in
                           range(meta['num_clusters'])}
    mutex_cluster = Lock()
    cluster_threadlist = []
    start_time = time.time()
    for i in range(meta['num_clusters']):
        cluster_id = i
        Q_mat_cluster = Q_mat[Q_mat_label_map[cluster_id], :]
        Q_mat_cluster = Q_mat_cluster[:, Q_mat_label_map[cluster_id]]
        cluster_worker = Thread(target=cluster_optim_qip_run_python_subroutine,
                                args=(cluster_id, Q_mat_cluster, Q_mat_label_map, input_filename, MQLib_dir, meta,
                                      output_filename, maxSetMap, maxRevMap,
                                      mutex_cluster))
        cluster_worker.start()
        cluster_threadlist.append(cluster_worker)

    for t in cluster_threadlist:
        t.join()

    time_taken = time.time() - start_time
    maxSet = np.concatenate(list(maxSetMap.values()))
    x_vec = np.zeros(num_variables, dtype=bool)
    for item in maxSet:
        x_vec[int(item) - 1] = 1
    Q_res = Q_mat[x_vec, :]
    Q_res = Q_res[:, x_vec]
    maxRev = np.sum(Q_res)

    # Convert To Old Indexing if required
    if ('is_improved_qubo' in meta.keys()):
        try:
            maxSet = [(new2old_index[i - 1] + 1) for i in maxSet]
        except:
            maxSet = []
            maxRev = 0

    return maxRev, maxSet, time_taken


def cluster_optim_qip_run_python_subroutine(cluster_id, Q_mat_cluster, Q_mat_label_map, input_filename, MQLib_dir, meta,
                                            output_filename, maxSetMap,
                                            maxRevMap, mutex_cluster):
    num_variables_cluster_id = Q_mat_cluster.shape[0]
    cluster_input_filename = f"{'/'.join(input_filename.split('/')[:-1])}/{cluster_id}_{input_filename.split('/')[-1]}"
    cluster_output_filename = \
        f"{'/'.join(output_filename.split('/')[:-1])}/{cluster_id}_{output_filename.split('/')[-1]}"
    with open(cluster_input_filename, 'w') as f:
        f.write(
            f'{num_variables_cluster_id} {int(num_variables_cluster_id * (num_variables_cluster_id + 1) / 2)}\n')
        for i in range(num_variables_cluster_id):
            for j in range(i, num_variables_cluster_id):
                f.write(f'{i + 1} {j + 1} {Q_mat_cluster[i][j]}\n')
    start_time = time.time()
    heuristic_list = meta['heuristic_list']
    time_limit = meta['time_multiplier'] * num_variables_cluster_id
    threadlist = []
    mutex = Lock()
    repeat_counter = 0
    maxRev, maxSet = -1, []
    while (repeat_counter <= meta['max_repeat_counter']) & (maxRev <= 0):
        # logger.info(f"Revenue: {maxRev}, repeat counter:{repeat_counter}")
        results = {}
        for i in range(len(heuristic_list)):
            worker = Thread(target=compare_qip_run_c_subroutine,
                            args=(
                                cluster_input_filename, MQLib_dir, heuristic_list[i],
                                (2 ** (repeat_counter)) * time_limit,
                                cluster_output_filename, results, mutex))
            threadlist.append(worker)
            worker.start()
        for t in threadlist:
            t.join()
        for heuristic in results.keys():
            if results[heuristic][0] > maxRev:
                maxRev, maxSet = results[heuristic][0], results[heuristic][1]
        repeat_counter += 1
    # logger.info(f"Got Results in {repeat_counter + 1} iterations...")
    time_taken = time.time() - start_time

    mutex_cluster.acquire()
    maxSetMap[cluster_id] = Q_mat_label_map[cluster_id][list(np.array(maxSet) - 1)]
    maxSetMap[cluster_id] = [xr + 1 for xr in maxSetMap[cluster_id]]
    maxRevMap[cluster_id] = maxRev
    mutex_cluster.release()
    return None


def binSearchCompare_qip_approx_multithread(num_prods, C, rcm, meta, K):
    time_log = {}
    start_time = time.time()
    maxcut_heuristic_list = ['BURER2002', 'FESTA2002G', 'FESTA2002GPR', 'FESTA2002VNS', 'FESTA2002VNSPR',
                             'FESTA2002GVNS',
                             'FESTA2002GVNSPR', 'DUARTE2005', 'LAGUNA2009CE', 'LAGUNA2009HCE']

    input_filename = meta['QIPApprox_input']
    output_filename = meta['QIPApprox_output']
    heuristic_list = meta['heuristic_list']
    time_limit = meta['time_multiplier'] * num_prods
    is_debug = meta['print_debug']
    MQLib_dir = meta['MQLib_dir']
    p_arr = rcm['p']
    v_arr = rcm['v']
    vij_arr = rcm['v2']
    is_improved_qubo = False
    max_repeat_counter = meta['max_repeat_counter']
    logger.info(f"time taken in setting up step variables {(time.time() - start_time) * 1e6} microsecs")
    time_log[f'variables_setup'] = (time.time() - start_time) * 1e6

    if ('is_improved_qubo' in meta.keys()):
        start_time = time.time()
        is_improved_qubo = True
        old2new_index, new2old_index = {}, {}
        selected_products = meta['selected_products']
        removed_products = meta['removed_products']
        not_allowed_products = selected_products + removed_products
        new_product_count = num_prods - len(not_allowed_products)
        logger.info(f"time taken in setting up extra improved variables {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'extra_improved_variables_setup'] = (time.time() - start_time) * 1e6
    else:
        selected_products = None
        removed_products = None
        not_allowed_products = None
        new_product_count = num_prods

    # Create Empty Q matrix with appropriate size
    start_time = time.time()
    num_variables = new_product_count
    constraints_allowed = False
    if 'constraints_allowed' in meta.keys():
        if (meta['constraints_allowed']) & ('max_assortment_size' in meta.keys()):
            constraints_allowed = True
    if constraints_allowed:
        num_variables += meta['max_assortment_size'] - 1

    Q_mat = np.zeros((num_variables, num_variables))
    logger.info(f"time taken in allocating empty Q matrix {(time.time() - start_time) * 1e6} microsecs")
    time_log[f'allocate_empty_Q_matrix'] = (time.time() - start_time) * 1e6

    if not is_improved_qubo:
        # Setup Input File for approx step
        start_time = time.time()
        for i in range(num_prods):
            for j in range(i, num_prods):
                Q_mat[i, j] = compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr, K)
        logger.info(f"time taken in filling Q matrix(unimproved) {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'fill_unimproved_Q_matrix'] = (time.time() - start_time) * 1e6
        # with open(input_filename, 'w') as f:
        #     f.write(f'{num_prods} {int(num_prods * (num_prods + 1) / 2)}\n')
        #     for i in range(num_prods):
        #         for j in range(i, num_prods):
        #             f.write(f'{i + 1} {j + 1} {compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr, K)}\n')
    else:
        # -------------
        # Improved Setup index transfer
        start_time = time.time()
        old_products = np.ones((num_prods + 1))
        old_products[not_allowed_products] = 0
        old_products = old_products[1:]
        new2old_index = np.where(old_products == 1)[0]
        logger.info(f"time taken in improved index assignment {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'index_assignment'] = (time.time() - start_time) * 1e6

        # SetupIndex transfer
        # start_time = time.time()
        # new_index = 0
        # for i in range(num_prods):
        #     if i + 1 in not_allowed_products:
        #         continue
        #     else:
        #         old2new_index[i] = new_index
        #         new_index += 1
        #
        # for key, val in old2new_index.items():
        #     new2old_index[val] = key
        #
        # logger.info(f"time taken in index assignment {(time.time() - start_time) * 1e6} microsecs")
        # time_log[f'index_assignment'] = (time.time() - start_time) * 1e6
        # -------------

        # -------------
        # ADD improved version in Index assignment
        start_time = time.time()
        for i in range(new_product_count):
            for j in range(i, new_product_count):
                idx_i, idx_j = new2old_index[i], new2old_index[j]
                if i == j:
                    Q_mat[i, j] = v_arr[idx_i + 1] * (p_arr[idx_i + 1] - K)
                    selection_add = [vij_arr[tuple([xr, idx_i + 1])] / 2 * (p_arr[xr] + p_arr[idx_i + 1] - K) for xr in
                                     selected_products]
                    Q_mat[i, j] += sum(selection_add)
                else:
                    Q_mat[i, j] = (vij_arr[tuple([idx_i + 1, idx_j + 1])] / 2) * (
                            p_arr[idx_i + 1] + p_arr[idx_j + 1] - K)
        logger.info(f"time taken in filling improved re-indexed Q matrix {(time.time() - start_time) * 1e6} microsecs")
        time_log[f'fill_reindexed_Q_matrix'] = (time.time() - start_time) * 1e6

        # Older version of Q_matrix
        # start_time = time.time()
        # for i in range(num_prods):
        #     if (i + 1) in removed_products:
        #         continue
        #     elif (i + 1) in selected_products:
        #         for j in range(i, num_prods):
        #             if (j + 1) not in not_allowed_products:
        #                 Q_mat[old2new_index[j], old2new_index[j]] += compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr,
        #                                                                                   K)
        #     else:  # i not in no optim products
        #         for j in range(i, num_prods):
        #             if (j + 1) in removed_products:
        #                 continue
        #             elif (j + 1) in selected_products:
        #                 Q_mat[old2new_index[i], old2new_index[i]] += compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr,
        #                                                                                   K)
        #             else:  # J not in no optim products
        #                 Q_mat[old2new_index[i], old2new_index[j]] += compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr,
        #                                                                                   K)
        # logger.info(f"time taken in filling re-indexed Q matrix {(time.time() - start_time) * 1e6} microsecs")
        # time_log[f'fill_reindexed_Q_matrix'] = (time.time() - start_time) * 1e6
        # -------------

    if constraints_allowed:
        start_time = time.time()
        penalty = 1e2
        # penalty = 100
        max_assortment_size = meta['max_assortment_size']
        for i in range(num_variables):
            for j in range(i, num_variables):
                if i == j:
                    Q_mat[i, j] = Q_mat[i, j] - (penalty * (1 - (2 * max_assortment_size)))
                else:
                    Q_mat[i, j] = Q_mat[i, j] - (penalty)
        penalty_constant = penalty * (max_assortment_size ** 2)
        logger.info(f"time taken in adding constraints in Q matrix {(time.time() - start_time) * 1e6} microsecs")

    # -------------
    # Improved Version of file write
    start_time = time.time()
    file_str_arr = ['%d %d %f\n' % (i + 1, j + 1, Q_mat[i][j]) for i in range(num_variables) for j in
                    range(i, num_variables)]
    file_str = f'{num_variables} {int(num_variables * (num_variables + 1) / 2)}\n' + ''.join(file_str_arr)
    open(input_filename, 'w').write(file_str)
    logger.info(f"time taken in writing Q matrix to file {(time.time() - start_time) * 1e6} microsecs")
    time_log[f'write_Q_matrix'] = (time.time() - start_time) * 1e6

    # start_time = time.time()
    # with open(input_filename, 'w') as f:
    #     f.write(f'{num_variables} {int(num_variables * (num_variables + 1) / 2)}\n')
    #     for i in range(num_variables):
    #         for j in range(i, num_variables):
    #             f.write(f'{i + 1} {j + 1} {Q_mat[i][j]}\n')
    # logger.info(f"time taken in writing Q matrix to file {(time.time() - start_time) * 1e6} microsecs")
    # time_log[f'write_Q_matrix'] = (time.time() - start_time) * 1e6
    # -------------

    start_time = time.time()
    heuristic_list = meta['heuristic_list']
    time_limit = meta['time_multiplier'] * new_product_count
    threadlist = []
    mutex = Lock()
    repeat_counter = 1
    maxRev, maxSet = -1, []
    logger.info(f"time taken in setting up threading mechanism {(time.time() - start_time) * 1e6} microsecs")
    time_log[f'setup_threading'] = (time.time() - start_time) * 1e6

    while ((repeat_counter <= max_repeat_counter) & (maxRev <= 0)):
        logger.info(f"Revenue: {maxRev}, repeat counter:{repeat_counter}")
        start_time_threads = time.time()
        results = {}
        for i in range(len(heuristic_list)):
            worker = Thread(target=compare_qip_run_c_subroutine,
                            args=(input_filename, MQLib_dir, heuristic_list[i], (2 ** (repeat_counter)) * time_limit,
                                  output_filename, results, mutex))
            threadlist.append(worker)
            worker.start()
        logger.info(f"time taken in setting up threads {(time.time() - start_time_threads) * 1e6} microsecs")
        time_log[f'R{repeat_counter}__starting_threads'] = (time.time() - start_time_threads) * 1e6

        start_time_threads = time.time()
        for t in threadlist:
            t.join()
        logger.info(f"time taken for threads to finish {(time.time() - start_time_threads) * 1e6} microsecs")
        time_log[f'R{repeat_counter}__finishing_threads'] = (time.time() - start_time_threads) * 1e6

        start_time_threads = time.time()
        for heuristic in results.keys():
            # Remove Slack Variables from maxSet
            maxSetheuristic = [xs for xs in results[heuristic][1] if xs <= new_product_count]
            if constraints_allowed:
                # maxRevheuristic = rcm_calc_revenue(maxSetheuristic, rcm['p'], rcm, num_prods)
                maxRevheuristic = results[heuristic][0] - penalty_constant
                if maxRevheuristic > maxRev:
                    maxRev, maxSet = maxRevheuristic, maxSetheuristic
            else:
                if results[heuristic][0] > maxRev:
                    maxRev, maxSet = results[heuristic][0], maxSetheuristic
        logger.info(f"time taken for get maxrev,maxset comparing "
                    f"heuristics {(time.time() - start_time_threads) * 1e6} microsecs")
        time_log[f'R{repeat_counter}__comparing_heuristic_results'] = (time.time() - start_time_threads) * 1e6

        repeat_counter += 1
    # logger.info(f"Got Results in {repeat_counter + 1} iterations...")
    time_taken = time.time() - start_time

    # Convert To Old Indexing if required
    start_time = time.time()
    if is_improved_qubo:
        try:
            maxSet = [(new2old_index[i - 1] + 1) for i in maxSet]
        except:
            maxSet = []
            maxRev = 0
    logger.info(f"time taken for reindex to old solution {(time.time() - start_time) * 1e6} microsecs")
    time_log[f'deindex_new_solution'] = (time.time() - start_time_threads) * 1e6
    return maxRev, maxSet, time_log


def compare_qip_run_c_subroutine(input_filename, MQLib_dir, heuristic, time_limit, output_filename, results, mutex):
    output_dir = '/'.join(output_filename.split("/")[:-1])
    output_file = output_filename.split("/")[-1]
    run_command = f'{MQLib_dir}/bin/MQLib -fQ {input_filename} -h {heuristic} -r {time_limit} -ps > {output_dir}/{heuristic}_{output_file}'
    os.system(run_command)
    try:
        with open(f'{output_dir}/{heuristic}_{output_file}') as f:
            summary_line = f.readline()
            line = summary_line
            i = 0
            while not (line[:-1] == 'Solution:'):
                line = f.readline()
                i += 1
                if i > 100:
                    break
            line = f.readline()
            solution = [int(x) for x in line[:-1].split(' ')]
        revSet = float(summary_line.split(",")[3])
        solSet = [(i + 1) for i in range(len(solution)) if (solution[i] == 1)]
    except:
        revSet, solSet = -1, []
    mutex.acquire()
    results[heuristic] = [revSet, solSet]
    mutex.release()
    return None


def compare_qip_get_Qval(i, j, p_arr, v_arr, vij_arr, K):
    if i == j:
        return v_arr[i + 1] * (p_arr[i + 1] - K)
    else:
        return (vij_arr[tuple([i + 1, j + 1])] / 2) * (p_arr[i + 1] + p_arr[j + 1] - K)


# --------------------BinSearch Compare Step: Linear Integer Programming -------------------

def binSearchCompare_ip_exact(num_prods, C, rcm, meta, K):
    logger.info("binSearchCompare_ip_exact Not implemented yet")
    return None


# --------------------BinSearch Compare Step: Nearest Neighbours -------------------

def binSearchCompare_nn(num_prods, C, rcm, meta, K):
    v = rcm['v']
    normConst = meta['normConst']
    db = meta['db']
    ptsTemp = meta['ptsTemp']

    vTemp_linear = np.concatenate((v[1:], -K * v[1:]))
    vTemp_quadratic = np.zeros(num_prods * (num_prods - 1))
    idxk = 0
    for idxi in range(num_prods):
        for idxj in range(idxi + 1, num_prods):
            vTemp_quadratic[idxk] = rcm['v2'][(idxi + 1, idxj + 1)]
            vTemp_quadratic[int(0.5 * num_prods * (num_prods - 1)) + idxk] = -K * rcm['v2'][(idxi + 1, idxj + 1)]
            idxk += 1
    vTemp = np.concatenate((vTemp_linear, vTemp_quadratic))
    query = np.concatenate(
        (vTemp, [0]))  # appending extra coordinate as recommended by Simple LSH, no normalization being done

    t_before = time.time()
    distList, approx_neighbors = db.kneighbors(query.reshape(1, -1), return_distance=True)

    nn_set = ast_from_set_char(ptsTemp[approx_neighbors[0][0], num_prods:2 * num_prods] * normConst)

    pseudoRev = np.linalg.norm(query) * (1 - distList) * normConst
    # realRev = rcm_calc_revenue(nn_set, rcm['p'], rcm, num_prods)
    queryTimeLog = None

    # logger.info('K:',K,' inner num_prods rev by v0:',pseudoRev/rcm['v'][0],' real rev:',realRev,' ast:',nn_set)

    return pseudoRev, nn_set, queryTimeLog


def compare_nn_get_pt_for_set(set_char_vector, C, p, normConst, num_prods):
    linear_part = np.concatenate((np.multiply(set_char_vector, p[1:]), set_char_vector))
    quadratic_part = np.zeros(num_prods * (num_prods - 1))
    idxk = 0
    for idxi in range(num_prods):
        for idxj in range(idxi + 1, num_prods):
            quadratic_part[idxk] = (p[idxi + 1] + p[idxj + 1]) * set_char_vector[idxi] * set_char_vector[idxj]
            quadratic_part[int(0.5 * num_prods * (num_prods - 1)) + idxk] = set_char_vector[idxi] * set_char_vector[
                idxj]
            idxk += 1

    return np.concatenate((linear_part, quadratic_part)) * 1.0 / normConst


def compare_nn_get_all_feasibles(num_prods, C, p):
    normConst = C * (C + 1) * np.sqrt(1 + 2 * np.max(p) ** 2)

    ptsTemp = []
    for size in range(1, C + 1):
        for ast in combinations(range(1, num_prods + 1), size):
            set_char_vector = set_char_from_ast(ast, num_prods)
            ptsTemp.append(compare_nn_get_pt_for_set(set_char_vector, C, p, normConst, num_prods))

    return np.array(ptsTemp), normConst


def compare_nn_preprocess(num_prods, C, p, algo, nEst=10, nCand=4):
    t0 = time.time()

    if algo == 'nn-approx':
        db = NearestNeighbors(n_estimators=nEst, n_candidates=nCand, n_neighbors=1, min_hash_match=2)
    elif algo == 'nn-exact':
        db = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
    else:
        db = None

    ptsTemp, normConst = compare_nn_get_all_feasibles(num_prods, C, p)

    # MIPS to NN transformation of all points
    lastCol = np.linalg.norm(ptsTemp, axis=1) ** 2
    lastCol = np.sqrt(1 - lastCol)
    pts = np.concatenate((ptsTemp, lastCol.reshape((ptsTemp.shape[0], 1))), axis=1)

    db.fit(pts)

    build_time = time.time() - t0
    logger.info("\t\tIndex build time: ", build_time)

    return db, build_time, normConst, ptsTemp


# ====================RCM Brute Search in all possible assortments===================

def rcm_brute_force_search(num_prods, C, rcm, meta=None, K=None):
    '''
        best assortment for ground truth model
    '''
    p = rcm['p']
    revs = []
    ast_list = []
    st = time.time()
    logger.info("Brute Force Method: Fixed Max Assortment Size to be less than 10")
    maxAssortSize = min(10, C)
    for size in range(1, maxAssortSize + 1):
        for ast in combinations(range(1, num_prods + 1), size):
            rev_temp = rcm_calc_revenue(ast, p, rcm, num_prods)
            revs.append(rev_temp)
            ast_list.append(ast)
    rev_best = np.array(revs).max()
    ast_best = ast_list[np.array(revs).argmax()]
    timeTaken = time.time() - st

    if meta is not None:
        if meta.get('print_results', False) is not False:
            logger.info('best rev:', rev_best, ' ast:', ast_best, ' time taken:', timeTaken)
    solve_log = {}
    return rev_best, ast_best, timeTaken, solve_log


# ====================RCM ADXOPT1 with products===================

def rcm_adxopt1_products(num_prods, C_old, rcm, meta=None):
    # potentially have a constraint on the assortment size
    C = num_prods
    if 'max_assortment_size' in meta.keys():
        C = meta['max_assortment_size']

    p = rcm['p']
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
    rev_prev = rcm_calc_revenue(set_prev, p, rcm, num_prods)

    rev_cal_counter = 0
    while True:
        items_left = [x for x in items if x not in set_prev]
        # Additions
        set_addition = []
        rev_addition = 0
        if len(set_prev) < C:
            for j in items_left:
                if removals[j] < b:
                    candidate_rev = rcm_calc_revenue(sorted(set_prev + [j]), p, rcm, num_prods)
                    rev_cal_counter += 1
                    if candidate_rev > rev_addition:
                        rev_addition = candidate_rev
                        set_addition = sorted(set_prev + [j])

        # Deletions
        set_deletion = []
        rev_deletion = 0
        if len(set_prev) > 0:
            for idx in range(len(set_prev)):
                candidate_rev = rcm_calc_revenue(sorted(set_prev[:idx] + set_prev[idx + 1:]), p, rcm, num_prods)
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
                        candidate_rev = rcm_calc_revenue(sorted(set_prev[:idx] + [j] + set_prev[idx + 1:]), p, rcm,
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

    # logger.info(set_current,p,rcm,num_prods)
    rev_adx = rcm_calc_revenue(set_current, p, rcm, num_prods)
    logger.info(f"\t\tNumber of times rcm_calc_revenue is called:{rev_cal_counter}")
    logger.info(f'rev adx:{rev_adx}')
    logger.info(f"\t\tProducts in the adxopt assortment are {set_current}")
    logger.info(f'\t\tTime taken for running adxopt is {timeTaken}')
    solve_log = {
        'count_calc_revenue_called': rev_cal_counter,
    }

    return rev_adx, set_current, timeTaken, solve_log


# ====================RCM ADXOPT2 with subsets===================
# This function considers addition of subsets of size 2 in addition to individual items
def rcm_adxopt2_sets(num_prods, C_old, rcm, meta=None, two_sets=True, b=None, allow_exchange=True):
    # potentially have a constraint on the assortment size
    C = num_prods
    if 'max_assortment_size' in meta.keys():
        C = meta['max_assortment_size']

    p = rcm['p']
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
    rev_prev = rcm_calc_revenue(set_prev, p, rcm, num_prods)

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
                    candidate_rev = rcm_calc_revenue(sorted(set_prev + list(j)), p, rcm, num_prods)
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
            #                 candidate_rev = rcm_calc_revenue(sorted(set_prev[:idx]+set_prev[idx+1:]),p,rcm,num_prods)
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
                candidate_rev = rcm_calc_revenue(sorted(list(np.setdiff1d(set_prev, subset))), p, rcm, num_prods)
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
                    #    candidate_rev = rcm_calc_revenue(sorted(set_prev[:idx]+[j]+set_prev[idx+1:]),p,rcm,num_prods)
                    for subset in avail_subsets:
                        new_ast = sorted(list(np.setdiff1d(set_prev, subset)) + list(j))
                        if len(new_ast) <= C:
                            candidate_rev = rcm_calc_revenue(new_ast, p, rcm, num_prods)
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

    rev_adx = rcm_calc_revenue(set_current, p, rcm, num_prods)
    if meta is not None:
        if meta.get('print_results', False) is not False:
            logger.info(f"\t\tNumber of times rcm_calc_revenue is called:{rev_cal_counter}")
            logger.info(f'rev adx:{rev_adx}')
            logger.info(f"\t\tProducts in the adxopt assortment are {set_current}")
            logger.info(f'\t\tTime taken for running adxopt is {timeTaken}')

    solve_log = {
        'count_calc_revenue_called': rev_cal_counter,
    }
    return rev_adx, set_current, timeTaken, solve_log


# ====================RCM Mixed Integer program ===================

def rcm_mixed_ip(num_prods, C, rcm, meta=None):
    # potentially have a constraint on the assortment size
    C = num_prods
    if 'max_assortment_size' in meta.keys():
        C = meta['max_assortment_size']

    opt_model = cpx.Model(name="rcm Model")
    # Declare Variables
    p_vars = \
        {(i, j): opt_model.continuous_var(lb=0, ub=1,
                                          name=f"p_{i}_{j}") for i in range(1, num_prods + 1) for j in
         range(i, num_prods + 1)}
    p_vars.update({(0, 0): opt_model.continuous_var(lb=0, ub=1, name=f"p_{0}_{0}")})

    x_vars = \
        {(i, j): opt_model.binary_var(name=f"x_{i}_{j}") for i in range(1, num_prods + 1) for j in
         range(i, num_prods + 1)}

    # Add Constraints to Model
    constraintCapacity = {1: opt_model.add_constraint(
        opt_model.sum(x_vars[i, i] for i in range(1, num_prods + 1)) <= C, ctname=f"constraintCapacity")}

    constraintsA = {(i, j): opt_model.add_constraint(
        ct=p_vars[i, j] <= x_vars[i, j], ctname=f"constraint_p_{i}{j}<x{i}{j}") for i in range(1, num_prods + 1) for j
        in
        range(i, num_prods + 1)}

    constraintsB = {(i, j): opt_model.add_constraint(
        ct=p_vars[i, j] <= mixed_ip_get_v_var(rcm, i, j) * p_vars[0, 0], ctname=f"constraint_p_{i}{j}<=V{i}{j}p_{0}{0}")
        for i in
        range(1, num_prods + 1) for j in range(i, num_prods + 1)}

    constraintsC = {(i, j): opt_model.add_constraint(
        ct=p_vars[i, j] + (mixed_ip_get_v_var(rcm, i, j) * (1 - x_vars[i, j])) >= mixed_ip_get_v_var(rcm, i, j) *
           p_vars[0, 0],
        ctname=f"constraint_p2_{i}{j}+V{i}{j}(1-x{i}{j})>=V{i}{j}p1_{0}{0}") for i in range(1, num_prods + 1) for j in
        range(i, num_prods + 1)}

    constraintsD = {
        (i, j): opt_model.add_constraint(ct=x_vars[i, j] <= x_vars[i, i], ctname=f"constraint_x{i}{j}<=x{i}{i}") for i
        in range(1, num_prods + 1) for j in range(i, num_prods + 1)}
    constraintsE = {
        (i, j): opt_model.add_constraint(ct=x_vars[i, j] <= x_vars[j, j], ctname=f"constraint_x{i}{j}<=x{j}{j}") for i
        in range(1, num_prods + 1) for j in range(i, num_prods + 1)}
    constraintsF = {(i, j): opt_model.add_constraint(ct=x_vars[i, j] >= x_vars[i, i] + x_vars[j, j] - 1,
                                                     ctname=f"constraint_x{i}{j}>=x{i}{i}+x{j}{j}-1") for i in
                    range(1, num_prods + 1) for j in range(i, num_prods + 1)}

    constraintsG = {2: opt_model.add_constraint(
        p_vars[0, 0] + opt_model.sum(
            p_vars[i, j] for i in range(1, num_prods + 1) for j in range(i, num_prods + 1)) == 1,
        ctname=f"constraint_SUMpij=1")}

    # Add Objective Function to the Model
    objective = opt_model.sum(
        mixed_ip_get_r_val(rcm, i, j) * p_vars[i, j] for i in range(1, num_prods + 1) for j in range(i, num_prods + 1))

    # Solve Optimization Model
    opt_model.maximize(objective)
    # TODO: Fixing Url and API Key here, Need to get it from meta
    # start_time = time.time()
    # if 'url' not in meta.keys():
    meta['url'] = "https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/"
    meta['key'] = "api_4e360631-3fd5-449b-b7a2-3449aca36a3b"

    # opt_model.solve(url=meta["url"], key=meta["key"])
    opt_model.solve()
    time_taken = opt_model.solution._solve_details._time
    # get Optimal Set and Optimal Revenue
    x_solution = {(i, j): x_vars[i, j].solution_value for i in range(1, num_prods + 1) for j in range(i, num_prods + 1)}
    maxRev = opt_model.objective_value
    maxSet = [x_key[0] for x_key in x_solution.keys() if x_solution[x_key] == 1 and x_key[0] == x_key[1]]
    solve_log = {}
    return maxRev, maxSet, time_taken, solve_log


def mixed_ip_get_v_var(rcm, i, j):
    if i == j:
        return rcm['v'][i] / rcm['v'][0]
    else:
        return rcm['v2'][tuple([i, j])] / rcm['v'][0]


def mixed_ip_get_r_val(rcm, i, j):
    if i == j:
        return rcm['p'][i]
    return rcm['p'][i] + rcm['p'][j]


# ===================Three Choice Model(TCM) Bonmin Solver==========

def tcm_bonmin_mnlip(num_prods, C, rcm, meta=None):
    # Write Data File for corresponding model
    bonmin_write_model_data_file(rcm, meta)
    data_filepath = meta['data_filepath']
    # create an instance
    st = time.time()
    instance = tcm_model.create_instance(data_filepath)
    results = opt.solve(instance, tee=True)
    maxSet = []
    for i in instance.x:
        if instance.x[i].value == 1.:
            maxSet.append(i)
    maxRev = instance.revenue.expr()
    timeTaken = time.time() - st
    solve_log = {}
    return maxRev, maxSet, timeTaken, solve_log


def bonmin_write_model_data_file(rcm, meta):
    prices = rcm['p'][1:]  # removing p0
    n = len(rcm['v']) - 1
    # open file handle
    filename = meta['data_filepath']
    with open(filename, 'w') as f:
        # Write N, v0
        f.write(f"param N := {n};\n")
        f.write(f"param v0 := {rcm['v'][0]};\n")

        # write prices
        f.write(f"param r :=\n")
        for i, price in enumerate(prices):
            f.write(f"{i + 1} {price}\n")
        f.write(";\n")

        # write v1s
        f.write(f"param v1 :=\n")
        for i in range(n):
            f.write(f"{i + 1} {rcm['v'][i + 1]}\n")
        f.write(";\n")

        # write v2s
        f.write(f"param v2 :=\n")
        for i in range(n):
            for j in range(i + 1, n):
                val = rcm['v2'][tuple([i + 1, j + 1])]
                f.write(f"{i + 1} {j + 1} {val}\n")
                f.write(f"{j + 1} {i + 1} {val}\n")
        f.write(";\n")

        # write v3s
        f.write(f"param v3 :=\n")
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    val = rcm['v3'][tuple([i + 1, j + 1, k + 1])]
                    f.write(f"{i + 1} {j + 1} {k + 1} {val}\n")
                    f.write(f"{i + 1} {k + 1} {j + 1} {val}\n")
                    f.write(f"{j + 1} {i + 1} {k + 1} {val}\n")
                    f.write(f"{j + 1} {k + 1} {i + 1} {val}\n")
                    f.write(f"{k + 1} {j + 1} {i + 1} {val}\n")
                    f.write(f"{k + 1} {i + 1} {j + 1} {val}\n")
        f.write(";\n")

    return


# ==================== Auxiliary Functions ===================
def rcm_get_assortment_probs(given_set, rcm, num_prods):
    # new indices for given_set for ease of computation
    set2idx = {}
    idx2set = {}
    for idx, x in enumerate(given_set):
        set2idx[x] = idx + 1
        idx2set[idx + 1] = x
    # logger.info(set2idx,idx2set)

    # Obtain probabilities, no choice only captured in normalization
    probs = {}
    normalization_temp = rcm['v'][0]
    # logger.info(normalization_temp)
    # single set choice
    if len(given_set) > 0:
        for ast in combinations(range(1, len(given_set) + 1), 1):
            set_char_vector = set_char_from_ast(idx2set[ast[0]], num_prods)
            probs[set_char_vector] = rcm['v'][idx2set[ast[0]]]
            normalization_temp += probs[set_char_vector]
        # logger.info(normalization_temp)
        # logger.info(ast,rcm['v'][idx2set[ast[0]]])
    # two set choice
    if len(given_set) > 1:
        for ast in combinations(range(1, len(given_set) + 1), 2):
            set_char_vector = set_char_from_ast([idx2set[x] for x in ast], num_prods)
            probs[set_char_vector] = rcm['v2'][(idx2set[ast[0]], idx2set[ast[1]])]
            normalization_temp += probs[set_char_vector]
        # logger.info(normalization_temp)
        # logger.info(ast,rcm['v2'][(ast[0],ast[1])])

    for x in probs:
        probs[x] /= normalization_temp

    return probs, normalization_temp


def rcm_calc_revenue_old(given_set, p, rcm, num_prods):
    probs, _ = rcm_get_assortment_probs(given_set, rcm, num_prods)
    rev = 0
    for set_char_vector in probs:
        subset_price = 0
        for i in range(len(set_char_vector)):
            if set_char_vector[i] == 1:
                subset_price += p[i + 1]
        rev = rev + subset_price * probs[set_char_vector]
    return rev


def rcm_calc_revenue(given_set, p, rcm, num_prods):
    if len(given_set) <= 0:
        return 0
    else:
        den0 = rcm['v'][0]
        num1 = np.sum([rcm['p'][xr] * rcm['v'][xr] for xr in given_set])
        den1 = np.sum([rcm['v'][xr] for xr in given_set])
        num2 = np.sum(
            [(rcm['p'][given_set[xi]] + rcm['p'][given_set[xj]]) * (rcm['v2'][tuple([given_set[xi], given_set[xj]])])
             for xi in range(len(given_set)) for xj in range(xi + 1, len(given_set))])
        den2 = np.sum([(rcm['v2'][tuple([given_set[xi], given_set[xj]])])
                       for xi in range(len(given_set)) for xj in range(xi + 1, len(given_set))])
        return (num1 + num2) / (den0 + den1 + den2)


def tcm_calc_revenue(subset, tcm):
    rev_num, rev_den = 0, tcm['v'][0]
    # 1 interactions
    if (len(subset) > 0):
        rev_num += sum([(tcm['v'][i] * tcm['p'][i]) for i in subset])
        rev_den += sum([(tcm['v'][i]) for i in subset])
    # 2 interactions
    if (len(subset) > 1):
        rev_num += sum([(tcm['v2'][tuple([i, j])]) * (tcm['p'][i] + tcm['p'][j]) for i in subset for j in subset if
                        (not i == j)]) * 0.5
        rev_den += sum([(tcm['v2'][tuple([i, j])]) for i in subset for j in subset if (not i == j)]) * 0.5

    # 3 interactions
    if (len(subset) > 2):
        rev_num += sum(
            [(tcm['v3'][tuple([i, j, k])]) * (tcm['p'][i] + tcm['p'][j] + tcm['p'][k]) for i in subset for j in subset
             for
             k in subset if ((not i == j) & (not j == k) & (not i == k))]) * (1 / 6)
        rev_den += sum([(tcm['v3'][tuple([i, j, k])]) for i in subset for j in subset for k in subset if
                        ((not i == j) & (not j == k) & (not i == k))]) * (1 / 6)
    print(rev_num, rev_den)
    return rev_num / rev_den

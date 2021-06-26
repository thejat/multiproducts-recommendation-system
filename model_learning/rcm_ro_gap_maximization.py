import numpy as np
from numpy import linalg
import itertools
import os
from scipy.optimize import minimize as Optimizer
from optimization.rcm_optim import rcm_binary_search_v2, rcm_revenue_ordered
from functools import partial


def get_optimality_gap(num_prod, p_arr, v_arr):
    v2 = dict()
    for i in range(1, num_prod + 1):
        for j in range(1, i):
            v2[tuple([i, j])] = v_arr[np.sum(range(i)) + j]
            v2[tuple([j, i])] = v2[tuple([i, j])]
    v1 = np.array([v_arr[np.sum(range(i)) + i] for i in range(1, num_prod + 1)])
    v1 = np.concatenate([[v_arr[0]], v1])
    p = np.insert(p_arr, 0, 0)
    rcm_model = {'v': v1, 'v2': v2, 'p': p}
    meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}
    meta.update({
        'num_prod'           : num_prod,
        'is_improved_qip'    : True,
        'comparison_function': 'qip-exact',
        'solution_id'        : 'binSearchImproved_qip_exact',
        'algo'               : 'binary-search-improved'
    })
    rev_best, _, _, _ = rcm_binary_search_v2(num_prod, num_prod, rcm_model, meta)
    rev_ro, _, _, _ = rcm_revenue_ordered(num_prod, num_prod, rcm_model, meta)
    return -1 * (rev_best - rev_ro) * 100 / rev_best


def maximize_ro_gap(num_prod, p_arr):
    v_arr_size = int(((num_prod + 1) * num_prod / 2) + 1)
    v_arr_0 = np.random.randomint(0, 100, v_arr_size)
    f_opt = partial(get_optimality_gap, num_prod, p_arr)
    res = Optimizer(fun=f_opt, x0=v_arr_0, method='nelder-mead', options={'fatol': 1e-8, 'disp': True})
    return res.x, f_opt(res.x)


dir = '../results/rev_ro_gap'
if not os.path.exists(dir):
    os.makedirs(dir)

# p = [786, 740, 794, 274, 711]
# v = np.array([1, 10, 59, 2, 8, 20, 90, 16, 8, 21, 44, 16, 62, 14, 73, 43]) / 100
# print(get_optimality_gap(5, p, v))
# num_prod = 5
for num_prod in range(10,50,10):
    p = np.random.randint(0,1e3*num_prod,num_prod)
    print('p',p)
    arr_size = int(((num_prod + 1) * num_prod / 2) + 1)
    max_gap, max_arr = 0, np.zeros(arr_size)
    with open(f'{dir}/ro_gap_{num_prod}.txt','a+') as f:
        f.write('num_prod|max_gap| p_arr|v_arr\n')
        for i in range(1000):
            curr_arr = np.random.randint(0,1e2*num_prod,arr_size)
            curr_gap = get_optimality_gap(num_prod, p, curr_arr)
            if(curr_gap<max_gap):
                max_gap, max_arr = curr_gap, curr_arr
                print(f"max_gap_iteration:{i}:{-1*max_gap}")
                f.write(f'{num_prod} | {str(round(max_gap,5))} | {max_arr.tolist()} | {p.tolist()}\n')

{'v' : [0.01, 0.1, 0.02, 0.9, 0.44, 0.43],
 'v2': {(2, 1): 0.59, (1, 2): 0.59, (3, 1): 0.08, (1, 3): 0.08,
        (3, 2): 0.2, (2, 3): 0.2, (4, 1): 0.16, (1, 4): 0.16,
        (4, 2): 0.08, (2, 4): 0.08, (4, 3): 0.21, (3, 4): 0.21,
        (5, 1): 0.16, (1, 5): 0.16, (5, 2): 0.62, (2, 5): 0.62,
        (5, 3): 0.14, (3, 5): 0.14, (5, 4): 0.73, (4, 5): 0.73},
 'p' : [0, 786, 740, 794, 274, 711]}

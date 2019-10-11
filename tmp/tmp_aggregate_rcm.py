import numpy as np
import pandas as pd
import os
from optim_experiments.rcm_optim_experiments import dump_rcm_models, run_rcm_experiments_v2
from datetime import datetime
import pickle
from copy import deepcopy

# Algorithms List
BINSEARCH_NN_EXACT = {'solution_id': 'binSearch_nn-exact', 'algo': 'binary-search', 'comparison_function': 'nn-exact'}
BINSEARCH_NN_APPROX = {'solution_id': 'binSearch_nn-approx', 'algo': 'binary-search',
                       'comparison_function': 'nn-approx'}
BINSEARCH_QIP_EXACT = {'solution_id': 'binSearch_qip_exact', 'algo': 'binary-search',
                       'comparison_function': 'qip-exact'}
BINSEARCH_QIP_BURER = {'solution_id': 'binSearch_qip_approx', 'algo': 'binary-search',
                       'comparison_function': 'qip-approx',
                       'type': 'chquick', 'QIPApprox_input': 'tmp/ApproxQIP_input_2.txt',
                       'QIPApprox_output': 'tmp/ApproxQIP_output_2.txt', 'heuristic': 'BURER2002',
                       'time_multiplier': 0.0001}
BINSEARCH_QIP_MTHREAD = {'solution_id': 'binSearch_qip_approx_multithread', 'algo': 'binary-search',
                         'comparison_function': 'qip-approx-mthread', 'QIPApprox_input': 'tmp/ApproxQIP_input.txt',
                         'QIPApprox_output': 'tmp/ApproxQIP_output.txt',
                         'heuristic_list': ['BURER2002', 'DUARTE2005', 'LAGUNA2009HCE', 'FESTA2002VNS'],
                         'time_multiplier': 0.0001}

MIXED_IP = {'solution_id': 'mixed_ip', 'algo': 'mixed-ip'}
ADXOPT1_PRODUCTS = {'solution_id': 'adxopt_products', 'algo': 'adxopt1'}
ADXOPT2_SETS = {'solution_id': 'adxopt_sets', 'algo': 'adxopt2'}
BRUTE_FORCE = {'solution_id': 'search_all', 'algo': 'brute-force'}

price_range_list = [100, 1000, 10000]
default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}
# -------------------------------RCM Very Small Tests Run(Use All Algorithms)----------------------------

# Directories
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
sme_rcm_model_dir = f'results/rcm_optim/small_experiments/rcm_models'
sme_rcm_solution_dir = 'results/rcm_optim/small_experiments/rcm_solutions'
sme_rcm_summary_dir = 'results/rcm_optim/small_experiments'

# Products Range and Repeat Count
sme_num_prods = list(range(2, 40, 2))
sme_repeat_count = 5

algorithm_list = [BINSEARCH_NN_EXACT, BINSEARCH_NN_APPROX, BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, MIXED_IP,
                  ADXOPT1_PRODUCTS, ADXOPT2_SETS, BRUTE_FORCE]

experiment_summary = []
for price_range in price_range_list:
    for num_prods in sme_num_prods:
        for repeat_id in range(sme_repeat_count):
            for optim_algo_dict in algorithm_list:
                # try:
                meta = deepcopy(default_meta)
                meta.update(optim_algo_dict)
                model_solve_filepath = \
                    f'{sme_rcm_solution_dir}/rcm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{repeat_id}.pkl'
                if os.path.exists(model_solve_filepath):
                    with open(model_solve_filepath, 'rb') as f:
                        sol_dict = pickle.load(f)
                    print(f"Retrieved RCM Solution {model_solve_filepath}...")
                    del sol_dict['rcm_model']
                    experiment_summary.append(sol_dict)


df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
df_results.to_csv(f"tmp_small.csv", index=False)

print("Finished")

# -------------------------------RCM Small-Medium Size Tests Run(Use All Algorithms)----------------------------

# Directories
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
mde_rcm_model_dir = f'results/rcm_optim/medium_experiments/rcm_models'
mde_rcm_solution_dir = 'results/rcm_optim/medium_experiments/rcm_solutions'
mde_rcm_summary_dir = 'results/rcm_optim/medium_experiments'

# Products Range and Repeat Count
mde_num_prods = list(range(40, 500, 40))
mde_repeat_count = 5

experiment_summary2 = []
for price_range in price_range_list:
    for num_prods in mde_num_prods:
        for repeat_id in range(mde_repeat_count):
            for optim_algo_dict in algorithm_list:
                # try:
                meta = deepcopy(default_meta)
                meta.update(optim_algo_dict)
                model_solve_filepath = \
                    f'{mde_rcm_solution_dir}/rcm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{repeat_id}.pkl'
                if os.path.exists(model_solve_filepath):
                    with open(model_solve_filepath, 'rb') as f:
                        sol_dict = pickle.load(f)
                    print(f"Retrieved RCM Solution {model_solve_filepath}...")
                    del sol_dict['rcm_model']
                    experiment_summary2.append(sol_dict)

df_results2 = pd.DataFrame.from_dict(dict(enumerate(experiment_summary2)), orient='index')
df_results2.to_csv(f"tmp_med.csv", index=False)

print("Finished")

# -------------------------------RCM Large Size Tests Run(Use All Algorithms)----------------------------

# Directories
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
lge_rcm_model_dir = f'results/rcm_optim/large_experiments/rcm_models'
lge_rcm_solution_dir = 'results/rcm_optim/large_experiments/rcm_solutions'
lge_rcm_summary_dir = 'results/rcm_optim/large_experiments'

# Products Range and Repeat Count
lge_num_prods = list(range(500, 7501, 500))
lge_repeat_count = 5

experiment_summary3 = []
for price_range in price_range_list:
    for num_prods in lge_num_prods:
        for repeat_id in range(lge_repeat_count):
            for optim_algo_dict in algorithm_list:
                # try:
                meta = deepcopy(default_meta)
                meta.update(optim_algo_dict)
                model_solve_filepath = \
                    f'{lge_rcm_solution_dir}/rcm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{repeat_id}.pkl'
                if os.path.exists(model_solve_filepath):
                    with open(model_solve_filepath, 'rb') as f:
                        sol_dict = pickle.load(f)
                    print(f"Retrieved RCM Solution {model_solve_filepath}...")
                    del sol_dict['rcm_model']
                    experiment_summary3.append(sol_dict)

df_results3 = pd.DataFrame.from_dict(dict(enumerate(experiment_summary3)), orient='index')
df_results3.to_csv(f"tmp_large.csv", index=False)

print("Finished")

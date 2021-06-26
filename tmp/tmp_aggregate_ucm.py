import numpy as np
import pandas as pd
import os
from optim_experiments.ucm_optim_experiments import run_ucm_experiments_v2, dump_ucm_models
from datetime import datetime
import pickle
from copy import deepcopy

MIXED_IP = {'solution_id': 'mixed_ip', 'algo': 'mixed-ip'}
ADXOPT1_PRODUCTS = {'solution_id': 'adxopt_products', 'algo': 'adxopt1'}
ADXOPT2_SETS = {'solution_id': 'adxopt_sets', 'algo': 'adxopt2'}

price_range_list = [100]
Hset_multiplier_list = [0, 0.25, 0.5, 1]
default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}
# -------------------------------UCM Very Small Tests Run(Use All Algorithms)----------------------------

# Directories
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
sme_ucm_model_dir = f'results/ucm_optim/small_experiments/ucm_models'
sme_ucm_solution_dir = 'results/ucm_optim/small_experiments/ucm_solutions'
sme_ucm_summary_dir = 'results/ucm_optim/small_experiments'

# Products Range and Repeat Count
sme_num_prods = list(range(2, 40, 2))
sme_repeat_count = 3

algorithm_list = [MIXED_IP, ADXOPT1_PRODUCTS, ADXOPT2_SETS]

# algorithm_list = [MIXED_IP]
experiment_summary = []
for price_range in price_range_list:
    for num_prods in sme_num_prods:
        for Hset_multiplier in Hset_multiplier_list:
            n_Hset_count = int(num_prods * (num_prods - 1) * Hset_multiplier / 2)
            for repeat_id in range(sme_repeat_count):
                for optim_algo_dict in algorithm_list:
                    meta = deepcopy(default_meta)
                    meta.update(optim_algo_dict)
                    model_solve_filepath = \
                        f'{sme_ucm_solution_dir}/ucm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{n_Hset_count}_{repeat_id}.pkl'
                    if os.path.exists(model_solve_filepath):
                        with open(model_solve_filepath, 'rb') as f:
                            sol_dict = pickle.load(f)
                        print(f"Retrieved UCM Solution {model_solve_filepath}...")
                        del sol_dict['ucm_model']
                        experiment_summary.append(sol_dict)


df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
df_results.to_csv(f"tmp_ucm_small.csv", index=False)
#
# print("Finished")
#
# # -------------------------------UCM Small-Medium Size Tests Run(Use All Algorithms)----------------------------
#
# # Directories
# time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
mde_ucm_model_dir = f'results/ucm_optim/medium_experiments/ucm_models'
mde_ucm_solution_dir = 'results/ucm_optim/medium_experiments/ucm_solutions'
mde_ucm_summary_dir = 'results/ucm_optim/medium_experiments'

# Products Range and Repeat Count
mde_num_prods = list(range(40, 300, 40))
mde_repeat_count = 3

experiment_summary2 = []
for price_range in price_range_list:
    for num_prods in mde_num_prods:
        for Hset_multiplier in Hset_multiplier_list:
            n_Hset_count = int(num_prods * (num_prods - 1) * Hset_multiplier / 2)
            for repeat_id in range(mde_repeat_count):
                for optim_algo_dict in algorithm_list:
                    # try:
                    meta = deepcopy(default_meta)
                    meta.update(optim_algo_dict)
                    model_solve_filepath = \
                        f'{mde_ucm_solution_dir}/ucm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{n_Hset_count}_{repeat_id}.pkl'
                    if os.path.exists(model_solve_filepath):
                        with open(model_solve_filepath, 'rb') as f:
                            sol_dict = pickle.load(f)
                        print(f"Retrieved UCM Solution {model_solve_filepath}...")
                        del sol_dict['ucm_model']
                        experiment_summary2.append(sol_dict)

df_results2 = pd.DataFrame.from_dict(dict(enumerate(experiment_summary2)), orient='index')
df_results2.to_csv(f"tmp_ucm_med.csv", index=False)
#
# print("Finished")
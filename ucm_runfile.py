import numpy as np
import pandas as pd
import os
from optim_experiments.ucm_optim_experiments import dump_ucm_models, run_ucm_experiments_v2
from datetime import datetime

# Algorithms List
MIXED_IP = {'solution_id': 'mixed_ip', 'algo': 'mixed-ip'}
ADXOPT1_PRODUCTS = {'solution_id': 'adxopt_products', 'algo': 'adxopt1'}
ADXOPT2_SETS = {'solution_id': 'adxopt_sets', 'algo': 'adxopt2'}
# BRUTE_FORCE = {'solution_id': 'search_all', 'algo': 'brute-force'}

price_range_list = [100,1000,10000]
Hset_multiplier_list = [0, 0.25, 0.5, 1]
default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}
# -------------------------------UCM Very Small Tests Run(Use All Algorithms)----------------------------

# Directories
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
sme_ucm_model_dir = f'results/ucm_optim/small_experiments/ucm_models'
sme_ucm_solution_dir = 'results/ucm_optim/small_experiments/ucm_solutions'
sme_ucm_summary_dir = 'results/ucm_optim/small_experiments'

# Products Range and Repeat Count
sme_num_prods = list(range(5, 50, 5))
sme_repeat_count = 10

sme_algorithm_list = [MIXED_IP, ADXOPT1_PRODUCTS, ADXOPT2_SETS]

if not os.path.exists(sme_ucm_model_dir):
    dump_ucm_models(price_range_list, sme_num_prods,Hset_multiplier_list, sme_repeat_count, dump_dir=sme_ucm_model_dir)
experiment_summary = run_ucm_experiments_v2(model_dir=sme_ucm_model_dir,
                                            algorithm_list=sme_algorithm_list,
                                            meta_default=default_meta,
                                            price_range_list=price_range_list,
                                            prod_count_list=sme_num_prods,
                                            Hset_multiplier_list=Hset_multiplier_list,
                                            repeat_count=sme_repeat_count,
                                            output_dir=sme_ucm_solution_dir)

df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
df_results.to_csv(f"{sme_ucm_summary_dir}/solution_summary_{time_now}.csv", index=False)

print("Finished")

# -------------------------------UCM Small-Medium Size Tests Run(Use All Algorithms)----------------------------

# Directories
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
mde_ucm_model_dir = f'results/ucm_optim/medium_experiments/ucm_models'
mde_ucm_solution_dir = 'results/ucm_optim/medium_experiments/ucm_solutions'
mde_ucm_summary_dir = 'results/ucm_optim/medium_experiments'

# Products Range and Repeat Count
mde_num_prods = list(range(50, 500, 50))
mde_repeat_count = 10
mde_algorithm_list = [MIXED_IP, ADXOPT1_PRODUCTS, ADXOPT2_SETS]

if not os.path.exists(mde_ucm_model_dir):
    dump_ucm_models(price_range_list, mde_num_prods,Hset_multiplier_list, mde_repeat_count, dump_dir=mde_ucm_model_dir)

experiment_summary = run_ucm_experiments_v2(model_dir=mde_ucm_model_dir,
                                            algorithm_list=mde_algorithm_list,
                                            meta_default=default_meta,
                                            price_range_list=price_range_list,
                                            prod_count_list=mde_num_prods,
                                            Hset_multiplier_list=Hset_multiplier_list,
                                            repeat_count=mde_repeat_count,
                                            output_dir=mde_ucm_solution_dir)

df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
df_results.to_csv(f"{mde_ucm_summary_dir}/solution_summary_{time_now}.csv", index=False)

print("Finished")
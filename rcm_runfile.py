import numpy as np
import pandas as pd
import os
from optim_experiments.rcm_optim_experiments import dump_rcm_models, run_rcm_experiments_v2
from datetime import datetime


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
sme_rcm_model_dir = f'results/optim/small_experiments/rcm_models'
sme_rcm_solution_dir = 'results/optim/small_experiments/rcm_solutions'
sme_rcm_summary_dir = 'results/optim/small_experiments'

# Products Range and Repeat Count
sme_num_prods = list(range(2, 100, 2))
sme_repeat_count = 20

sme_algorithm_list = [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, MIXED_IP,
                      ADXOPT1_PRODUCTS, ADXOPT2_SETS]

if not os.path.exists(sme_rcm_model_dir):
    dump_rcm_models(price_range_list, sme_num_prods, sme_repeat_count, dump_dir=sme_rcm_model_dir)
experiment_summary = run_rcm_experiments_v2(model_dir=sme_rcm_model_dir,
                                            algorithm_list=sme_algorithm_list,
                                            meta_default=default_meta,
                                            price_range_list=price_range_list,
                                            prod_count_list=sme_num_prods,
                                            repeat_count=sme_repeat_count,
                                            output_dir=sme_rcm_solution_dir)

df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
df_results.to_csv(f"{sme_rcm_summary_dir}/solution_summary_{time_now}.csv", index=False)


print("Finished")

# -------------------------------RCM Small-Medium Size Tests Run(Use All Algorithms)----------------------------

# Directories
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
mde_rcm_model_dir = f'results/optim/medium_experiments/rcm_models'
mde_rcm_solution_dir = 'results/optim/medium_experiments/rcm_solutions'
mde_rcm_summary_dir = 'results/optim/medium_experiments'

# Products Range and Repeat Count
mde_num_prods = list(range(100, 500, 50))
mde_repeat_count = 20
mde_algorithm_list = [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, ADXOPT2_SETS, ADXOPT1_PRODUCTS]

if not os.path.exists(mde_rcm_model_dir):
    dump_rcm_models(price_range_list, mde_num_prods, mde_repeat_count, dump_dir=mde_rcm_model_dir)

experiment_summary = run_rcm_experiments_v2(model_dir=mde_rcm_model_dir,
                                            algorithm_list=mde_algorithm_list,
                                            meta_default=default_meta,
                                            price_range_list=price_range_list,
                                            prod_count_list=mde_num_prods,
                                            repeat_count=mde_repeat_count,
                                            output_dir=mde_rcm_solution_dir)

df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
df_results.to_csv(f"{mde_rcm_summary_dir}/solution_summary_{time_now}.csv", index=False)

print("Finished")

# -------------------------------RCM Large Size Tests Run(Use All Algorithms)----------------------------

# Directories
time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
lge_rcm_model_dir = f'results/optim/large_experiments/rcm_models'
lge_rcm_solution_dir = 'results/optim/large_experiments/rcm_solutions'
lge_rcm_summary_dir = 'results/optim/large_experiments'

# Products Range and Repeat Count
lge_num_prods = list(range(500, 5000, 500))
lge_repeat_count = 20

lge_algorithm_list = [BINSEARCH_QIP_MTHREAD]
if not os.path.exists(lge_rcm_model_dir):
    dump_rcm_models(price_range_list, lge_num_prods, lge_repeat_count, dump_dir=lge_rcm_model_dir)
experiment_summary = run_rcm_experiments_v2(model_dir=lge_rcm_model_dir,
                                            algorithm_list=lge_algorithm_list,
                                            meta_default=default_meta,
                                            price_range_list=[100],
                                            prod_count_list=lge_num_prods,
                                            repeat_count=lge_repeat_count,
                                            output_dir=lge_rcm_solution_dir)

df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
df_results.to_csv(f"{lge_rcm_summary_dir}/solution_summary_{time_now}.csv", index=False)

print("Finished")

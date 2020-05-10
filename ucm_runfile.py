import numpy as np
import pandas as pd
import os, sys
from optim_experiments.ucm_optim_experiments import dump_ucm_models, run_ucm_experiments_v2, dump_derived_ucm_models
from datetime import datetime
from ucm_config import *
import logging
init_logger()
logger = logging.getLogger(__name__)

if len(sys.argv) < 2:
    print("No Experiment Sets to perform, Exiting..")
    exit(0)

experiment_id_list = sys.argv[1:]
for experiment_id in experiment_id_list:
    model_dir = ucm_model_dir % (experiment_id)
    solution_dir = ucm_solution_dir % (experiment_id)
    summary_dir = ucm_summary_dir % (experiment_id)

    num_prods = experiment_set_dict[experiment_id]['num_prods']
    repeat_count = experiment_set_dict[experiment_id]['repeat_count']
    if 'parent_model_file' in experiment_set_dict[experiment_id].keys():
        parent_model_file = experiment_set_dict[experiment_id]['parent_model_file']
        price_range_list=None
        Hset_multiplier_list=None
        dump_derived_ucm_models(parent_model_file, num_prods, repeat_count, dump_dir=model_dir)
    else:
        parent_model_file=None
        price_range_list = experiment_set_dict[experiment_id]['price_range']
        Hset_multiplier_list = experiment_set_dict[experiment_id]['Hset_multiplier_list']
        dump_ucm_models(price_range_list, num_prods, Hset_multiplier_list, repeat_count, dump_dir=model_dir)

    algorithm_list = experiment_set_dict[experiment_id]['algorithm_list']
    experiment_summary = run_ucm_experiments_v2(model_dir=model_dir,
                                                algorithm_list=algorithm_list,
                                                meta_default=default_meta,
                                                price_range_list=price_range_list,
                                                parent_model_file=parent_model_file,
                                                prod_count_list=num_prods,
                                                Hset_multiplier_list=Hset_multiplier_list,
                                                repeat_count=repeat_count,
                                                output_dir=solution_dir)
    df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
    df_results.to_csv(f"{summary_dir}/solution_summary.csv", index=False,sep='|')
    print(f"-----Processed Experiment Set {experiment_id}--------")

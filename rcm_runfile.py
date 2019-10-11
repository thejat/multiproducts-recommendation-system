import numpy as np
import pandas as pd
import os, sys
from optim_experiments.rcm_optim_experiments import dump_rcm_models, run_rcm_experiments_v2
from datetime import datetime
from rcm_config import *




if len(sys.argv) < 2:
    print("No Experiment Sets to perform, Exiting..")
    exit(0)

experiment_id_list = sys.argv[1:]
for experiment_id in experiment_id_list:
    model_dir = rcm_model_dir%(experiment_id)
    solution_dir = rcm_solution_dir%(experiment_id)
    summary_dir = rcm_summary_dir%(experiment_id)

    num_prods = experiment_set_dict[experiment_id]['num_prods']
    repeat_count = experiment_set_dict[experiment_id]['repeat_count']
    price_range_list = experiment_set_dict[experiment_id]['price_range']

    if not os.path.exists(model_dir):
        dump_rcm_models(price_range_list, num_prods, repeat_count, dump_dir=model_dir)

    algorithm_list = experiment_set_dict[experiment_id]['algorithm_list']
    experiment_summary = run_rcm_experiments_v2(model_dir=model_dir,
                                            algorithm_list=algorithm_list,
                                            meta_default=default_meta,
                                            price_range_list=price_range_list,
                                            prod_count_list=num_prods,
                                            repeat_count=repeat_count,
                                            output_dir=solution_dir)
    df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
    df_results.to_csv(f"{summary_dir}/solution_summary.csv", index=False)
    print(f"-----Processed Experiment Set {experiment_id}--------")

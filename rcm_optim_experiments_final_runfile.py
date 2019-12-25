import numpy as np
import pandas as pd
import os, sys
from optim_experiments.rcm_optim_experiments import dump_rcm_models, run_rcm_experiments_v2, dump_derived_rcm_models
from datetime import datetime
from rcm_optim_experiments_final_config import *

init_logger()
logger = logging.getLogger(__name__)
logger.info(
    f"\n\n\n\n==================NEW RUN, Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}=====================")
if len(sys.argv) < 2:
    logger.error("No Experiment Sets to perform, Exiting..")
    exit(0)

model_dir = rcm_model_dir
solution_dir = rcm_solution_dir
summary_dir = rcm_summary_dir
experiment_id_list = sys.argv[1:]
if experiment_id_list[0] == 'all':
    experiment_id_list = list(experiment_set_dict.keys())
    logger.warning("Running on All configs provided in Examples...")
    logger.warning(f"Available Configs: {','.join(experiment_id_list)}")

for experiment_id in experiment_id_list:
    num_prods = experiment_set_dict[experiment_id]['num_prods']
    repeat_count = experiment_set_dict[experiment_id]['repeat_count']
    if 'parent_model_file' in experiment_set_dict[experiment_id].keys():
        parent_model_file = experiment_set_dict[experiment_id]['parent_model_file']
        price_range_list = None
    else:
        price_range_list = experiment_set_dict[experiment_id]['price_range']
        parent_model_file = None
    # set default meta parameters
    experiment_id_meta = dict(default_meta)
    experiment_id_meta.update(
        {key: experiment_set_dict[experiment_id][key] for key in experiment_set_dict[experiment_id].keys() if
         key not in basic_keys})

    if 'prob_v0' in experiment_set_dict[experiment_id].keys():
        prob_v0 = experiment_set_dict[experiment_id]['prob_v0']
    else:
        prob_v0 = None
    prob_v0_model_creation = None
    if 'parent_model_file' in experiment_set_dict[experiment_id].keys():
        is_mnl_model,is_tcm_model = False, False
        if 'is_mnl_model' in experiment_set_dict[experiment_id].keys():
            is_mnl_model = experiment_set_dict[experiment_id]['is_mnl_model']

        if 'is_tcm_model' in experiment_set_dict[experiment_id].keys():
            is_tcm_model = experiment_set_dict[experiment_id]['is_tcm_model']

        dump_derived_rcm_models(parent_model_file, num_prods, repeat_count, dump_dir=model_dir,
                                prob_v0=prob_v0_model_creation, is_mnl=is_mnl_model, is_tcm=is_tcm_model)
    else:
        dump_rcm_models(price_range_list, num_prods, repeat_count, dump_dir=model_dir, prob_v0=prob_v0_model_creation)

    algorithm_list = experiment_set_dict[experiment_id]['algorithm_list']
    experiment_summary = run_rcm_experiments_v2(model_dir=model_dir,
                                                algorithm_list=algorithm_list,
                                                meta_default=experiment_id_meta,
                                                price_range_list=price_range_list,
                                                parent_model_file=parent_model_file,
                                                prod_count_list=num_prods,
                                                repeat_count=repeat_count,
                                                output_dir=solution_dir,
                                                prob_v0=prob_v0)

    df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
    df_results.to_csv(f"{summary_dir}/{experiment_id}_solution_summary.csv", index=False, sep='|')
    logger.info(f"-----Processed Experiment Set {experiment_id}--------\n\n\n")

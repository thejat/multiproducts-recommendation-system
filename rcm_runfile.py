import numpy as np
import pandas as pd
import os, sys
from optim_experiments.rcm_optim_experiments import dump_rcm_models, run_rcm_experiments_v2, dump_derived_rcm_models
from datetime import datetime
from rcm_config import *

init_logger()
logger = logging.getLogger(__name__)
logger.info(
    f"\n\n\n\n==================NEW RUN, Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}=====================")
if len(sys.argv) < 2:
    logger.error("No Experiment Sets to perform, Exiting..")
    exit(0)

experiment_id_list = sys.argv[1:]
for experiment_id in experiment_id_list:
    model_dir = rcm_model_dir % (experiment_id)
    solution_dir = rcm_solution_dir % (experiment_id)
    summary_dir = rcm_summary_dir % (experiment_id)

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
    if 'parent_model_file' in experiment_set_dict[experiment_id].keys():
        dump_derived_rcm_models(parent_model_file, num_prods, repeat_count, dump_dir=model_dir, prob_v0=prob_v0)
    else:
        dump_rcm_models(price_range_list, num_prods, repeat_count, dump_dir=model_dir, prob_v0=prob_v0)

    algorithm_list = experiment_set_dict[experiment_id]['algorithm_list']
    experiment_summary = run_rcm_experiments_v2(model_dir=model_dir,
                                                algorithm_list=algorithm_list,
                                                meta_default=experiment_id_meta,
                                                price_range_list=price_range_list,
                                                parent_model_file=parent_model_file,
                                                prod_count_list=num_prods,
                                                repeat_count=repeat_count,
                                                output_dir=solution_dir)

    if 'test_only' in experiment_set_dict[experiment_id]:
        if (experiment_set_dict[experiment_id]['test_only']):
            # remove solution directory
            os.system(f"rm -r {rcm_solution_dir % (experiment_id)}")
    else:  # Write Summary File
        df_results = pd.DataFrame.from_dict(dict(enumerate(experiment_summary)), orient='index')
        df_results.to_csv(f"{summary_dir}/solution_summary.csv", index=False)

    logger.info(f"-----Processed Experiment Set {experiment_id}--------\n\n\n")

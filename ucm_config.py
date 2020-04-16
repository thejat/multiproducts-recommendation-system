from algorithms_config import *
import os
import json
import logging, logging.config
import collections

default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}

ucm_model_dir = f'results/ucm_optim/%s_experiments/ucm_models'
ucm_solution_dir = 'results/ucm_optim/%s_experiments/ucm_solutions'
ucm_summary_dir = 'results/ucm_optim/%s_experiments'
RC = 50
small_lb = 20
small_ub = 100  # excludes ub
small_delta = 20
medium_lb = small_ub
medium_ub = 500
medium_delta = 100
large_lb = medium_ub
large_ub = 2000
large_delta = 500
small_array = list(range(small_lb, small_ub, small_delta))
medium_array = list(range(medium_lb, medium_ub, medium_delta))
large_array = list(range(large_lb, large_ub, large_delta))

experiment_set_dict = collections.OrderedDict({
    # Small Experiments Config
    'small': {
        'price_range': [1000],
        'num_prods': small_array,
        'repeat_count': RC,
        'Hset_multiplier_list': [0, 0.25, 0.5, 1],
        'algorithm_list': [MIXED_IP, ADXOPT1_PRODUCTS, ADXOPT2_SETS, REVENUE_ORDERED]
    },
    # Medium Experiments Config
    'medium': {
        'price_range': [1000],
        'num_prods': medium_array,
        'repeat_count': RC,
        'Hset_multiplier_list': [0.25],
        'algorithm_list': [MIXED_IP, ADXOPT1_PRODUCTS, REVENUE_ORDERED]
    },
    'uci_all':{
        'parent_model_file': 'synthetic_models/models/ucm_uci.pkl',
        'num_prods'           : small_array+medium_array,
        'repeat_count'        : RC,
        'Hset_multiplier_list': [0.25],
        'algorithm_list'      : [ADXOPT1_PRODUCTS, MIXED_IP, REVENUE_ORDERED]
    },
    'tafeng_all': {
        'parent_model_file'   : 'synthetic_models/models/ucm_tafeng.pkl',
        'num_prods'           : small_array,
        'repeat_count'        : RC,
        'Hset_multiplier_list': [0.25],
        'algorithm_list'      : [ADXOPT1_PRODUCTS, MIXED_IP, REVENUE_ORDERED]
    }
})


def init_logger(default_path='logging.json', default_level=logging.DEBUG, env_key='LOG_CFG', log_dir='results/logs/'):
    path = default_path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    value = os.getenv(env_key, None)

    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return None

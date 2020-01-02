from algorithms_config import *
import os
import json
import logging, logging.config
import collections

default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}

ucm_model_dir = f'results/ucm_optim/%s_experiments/ucm_models'
ucm_solution_dir = 'results/ucm_optim/%s_experiments/ucm_solutions'
ucm_summary_dir = 'results/ucm_optim/%s_experiments'

experiment_set_dict = collections.OrderedDict({
    # Small Experiments Config
    'small': {
        'price_range': [1000],
        'num_prods': list(range(20, 101, 20)),
        'repeat_count': 1,
        'Hset_multiplier_list': [0, 0.25, 0.5, 1],
        'algorithm_list': [MIXED_IP, ADXOPT1_PRODUCTS, ADXOPT2_SETS, REVENUE_ORDERED]
    },
    # Medium Experiments Config
    'medium': {
        'price_range': [1000],
        'num_prods': list(range(100, 500, 100)),
        'repeat_count': 50,
        'Hset_multiplier_list': [0.25],
        'algorithm_list': [MIXED_IP, ADXOPT1_PRODUCTS, REVENUE_ORDERED]
    },
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

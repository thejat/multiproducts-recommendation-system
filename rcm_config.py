from algorithms_config import *
import os
import json
import logging, logging.config

default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}
price_range_list = [100, 1000, 10000]
basic_keys = ['price_range', 'num_prods', 'repeat_count', 'algorithm_list']
rcm_model_dir = f'results/rcm_optim/%s_experiments/rcm_models'
rcm_solution_dir = 'results/rcm_optim/%s_experiments/rcm_solutions'
rcm_summary_dir = 'results/rcm_optim/%s_experiments'

experiment_set_dict = {
    'spc_test': {
        'price_range': [10000],
        'num_prods': [2000, 2500, 3000],
        'repeat_count': 1,
        'prob_v0': None,
        'test_only': True,
        'max_problem_size': 1000,
        'algorithm_list': [BINSEARCH_QIP_MTHREAD_SPC, BINSEARCHIMPROVED_QIP_MTHREAD]
    },

    'synthetic_model_test': {
        'num_prods': range(500, 1000, 50),
        'repeat_count': 1,
        'prob_v0': None,
        'max_assortment_size': 5,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'test_only': True,
        'algorithm_list': [BINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    # Revenue ordered assortments test
    'constrained_qubo_test': {
        'price_range': [100],
        'num_prods': [100],
        'repeat_count': 1,
        'prob_v0': None,
        'max_assortment_size': 5,
        'test_only': True,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, BINSEARCH_QIP_MTHREAD_CONSTRAINED,
                           BINSEARCHIMPROVED_QIP_MTHREAD]
    },
    # Revenue ordered assortments test
    'time_multiplier_test': {
        'price_range': [10000],
        'num_prods': [2000],
        'repeat_count': 1,
        'prob_v0': None,
        'test_only': True,
        'max_problem_size': 1000,
        'algorithm_list': [BINSEARCH_QIP_MTHREAD_SPC]
    },
    # Revenue ordered assortments test
    'revenue_ordered_test': {
        'price_range': [100],
        'num_prods': [20, 30, 40, 50],
        'repeat_count': 1,
        'prob_v0': None,
        'algorithm_list': [REVENUE_ORDERED, BINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCH_QIP_MTHREAD]
    },
    # V0 Config Set for generating Improved Binary Set
    'custom_v0_test': {
        'price_range': [100],
        'num_prods': [20],
        'repeat_count': 1,
        'prob_v0': 0.4,
        'algorithm_list': [MIXED_IP, BINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCH_QIP_MTHREAD]
    },
    # DevQA Config For Improved Binary Set
    'devqa_binary_search_improved': {
        'price_range': [100],
        'num_prods': list(range(20, 380, 20)),
        'repeat_count': 3,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCH_QIP_MTHREAD]
    },
    # Small Experiments Config
    'small': {
        'price_range': [100, 1000, 10000],
        'num_prods': list(range(5, 25, 5)),
        'repeat_count': 2,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, MIXED_IP,
                           ADXOPT1_PRODUCTS, ADXOPT2_SETS]
    },
    # Medium Experiments Config
    'medium': {
        'price_range': [100, 1000, 10000],
        'num_prods': list(range(100, 500, 50)),
        'repeat_count': 20,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, ADXOPT2_SETS, ADXOPT1_PRODUCTS]
    },
    # Large Experiments Config
    'large': {
        'price_range': [100, 1000, 10000],
        'num_prods': list(range(500, 5000, 500)),
        'repeat_count': 20,
        'algorithm_list': [BINSEARCH_QIP_MTHREAD]
    }
}


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

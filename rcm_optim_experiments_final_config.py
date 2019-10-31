from algorithms_config import *
import os
import json
import logging, logging.config

default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}
basic_keys = ['price_range', 'num_prods', 'repeat_count', 'algorithm_list']
rcm_model_dir = f'results/final_paper/rcm/models'
rcm_solution_dir = 'results/final_paper/rcm/solutions'
rcm_summary_dir = 'results/final_paper/rcm'

experiment_set_dict = {
    # Small Experiments Config
    'small': {
        'price_range': [1000],
        'num_prods': list(range(20, 100, 20)),
        'repeat_count': 2,
        'algorithm_list': [ADXOPT1_PRODUCTS, ADXOPT2_SETS, MIXED_IP,
                           BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                           BINSEARCHIMPROVED_QIP_MTHREAD]
    },
    # Medium Experiments Config
    'medium': {
        'price_range': [1000],
        'num_prods': list(range(100, 500, 100)),
        'repeat_count': 2,
        'algorithm_list': [ADXOPT1_PRODUCTS, MIXED_IP,
                           BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                           BINSEARCHIMPROVED_QIP_MTHREAD]
    },
    # Large Experiments Config
    'large': {
        'price_range': [1000],
        'num_prods': list(range(500, 2501, 500)),
        'repeat_count': 1,
        'algorithm_list': [BINSEARCH_QIP_MTHREAD_SPC, BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD,
                           BINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_MTHREAD_SPC]
    },
    'tafeng_small': {
        'num_prods': list(range(20, 100, 20)),
        'repeat_count': 2,
        'prob_v0': None,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list': [ADXOPT1_PRODUCTS, ADXOPT2_SETS, MIXED_IP,
                           BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                           BINSEARCHIMPROVED_QIP_MTHREAD]
    },
    'tafeng_medium': {
        'num_prods': list(range(100, 500, 100)),
        'repeat_count': 2,
        'prob_v0': None,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list': [ADXOPT1_PRODUCTS, MIXED_IP,
                           BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                           BINSEARCHIMPROVED_QIP_MTHREAD]
    },
    'tafeng_large': {
        'num_prods': list(range(500, 2501, 500)),
        'repeat_count': 2,
        'prob_v0': None,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_MTHREAD,
                           BINSEARCH_QIP_MTHREAD_SPC, BINSEARCHIMPROVED_QIP_MTHREAD_SPC]
    },
    'uci_small': {
        'num_prods': list(range(20, 100, 20)),
        'repeat_count': 2,
        'prob_v0': None,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list': [ADXOPT1_PRODUCTS, ADXOPT2_SETS, MIXED_IP,
                           BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                           BINSEARCHIMPROVED_QIP_MTHREAD]
    },
    'uci_medium': {
        'num_prods': list(range(100, 500, 100)),
        'repeat_count': 2,
        'prob_v0': None,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list': [ADXOPT1_PRODUCTS, MIXED_IP,
                           BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                           BINSEARCHIMPROVED_QIP_MTHREAD]
    },
    'uci_large': {
        'num_prods': list(range(500, 2501, 500)),
        'repeat_count': 2,
        'prob_v0': None,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_MTHREAD,
                           BINSEARCH_QIP_MTHREAD_SPC, BINSEARCHIMPROVED_QIP_MTHREAD_SPC]
    },
    'constrained_small': {
        'price_range': [1000],
        'num_prods': list(range(20, 100, 20)),
        'repeat_count': 2,
        'prob_v0': None,
        'max_assortment_size': 5,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    'constrained_medium_large': {
        'price_range': [1000],
        'num_prods': list(range(100, 500, 100)) + list(range(500, 2500, 500)),
        'repeat_count': 2,
        'prob_v0': None,
        'max_assortment_size': 20,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    'constrained_tafeng_small': {
        'num_prods': list(range(20, 100, 20)),
        'repeat_count': 2,
        'prob_v0': None,
        'max_assortment_size': 5,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    'constrained_tafeng_medium_large': {
        'num_prods': list(range(100, 500, 100)) + list(range(500, 2500, 500)),
        'repeat_count': 2,
        'prob_v0': None,
        'max_assortment_size': 20,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    'constrained_uci_small': {
        'num_prods': list(range(20, 100, 20)),
        'repeat_count': 2,
        'prob_v0': None,
        'max_assortment_size': 5,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    'constrained_uci_medium_large': {
        'num_prods': list(range(100, 500, 100)) + list(range(500, 2500, 500)),
        'repeat_count': 2,
        'prob_v0': None,
        'max_assortment_size': 20,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    # 'mnl_revenue_ordered': {
    #     'price_range': [1000],
    #     'num_prods': list(range(20, 100, 20)) + list(range(100, 500, 100)) + list(range(500, 2500, 500)),
    #     'repeat_count': 1,
    #     'prob_v0': None,
    #     'algorithm_list': [MNL_REVENUE_ORDERED, BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD]
    # }
}

v0_levels = range(10, 100, 20)
v0_configs = {
    f'v0_{xr}_synthetic': {
        'price_range': [1000],
        'num_prods': list(range(100, 500, 100)),
        'repeat_count': 2,
        'prob_v0': xr / 100,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_MTHREAD]
    }
    for xr in v0_levels
}

experiment_set_dict.update(v0_configs)


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

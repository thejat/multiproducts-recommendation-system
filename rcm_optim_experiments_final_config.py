from algorithms_config import *
import os
import json
import logging, logging.config
import collections

default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}
basic_keys = ['price_range', 'num_prods', 'repeat_count', 'algorithm_list']
rcm_model_dir = f'results/jun2021_iteration/rcm/models'
rcm_solution_dir = 'results/jun2021_iteration/rcm/solutions'
rcm_timelog_dir = 'results/jun2021_iteration/rcm/time_logs'
rcm_summary_dir = 'results/jun2021_iteration/rcm'
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

jun21_experiment_set_dict  =collections.OrderedDict({
    'synthetic_model_test': {
        'num_prods': list(range(50, 100, 50)),
        'repeat_count': 1,
        'prob_v0': None,
        'max_assortment_size': 5,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'test_only': True,
        'algorithm_list': [NOISYBINSEARCHIMPROVED_QIP_MTHREAD]
    },
})

experiment_set_dict = collections.OrderedDict({
    'small'                   : {
        'price_range'   : [1000],
        'num_prods'     : small_array,
        'repeat_count'  : RC,
        'algorithm_list': [ADXOPT1_PRODUCTS, MIXED_IP,
                           BINSEARCH_QIP_EXACT, NOISYBINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                           NOISYBINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_EXACT]
    },
    'tafeng_small'            : {
        'num_prods'        : small_array,
        'repeat_count'     : RC,
        'prob_v0'          : None,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list'   : [ADXOPT1_PRODUCTS, ADXOPT2_SETS, MIXED_IP,
                              BINSEARCH_QIP_EXACT, NOISYBINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                              NOISYBINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_EXACT]
    },
    'uci_small'               : {
        'num_prods'        : small_array,
        'repeat_count'     : RC,
        'prob_v0'          : None,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list'   : [ADXOPT1_PRODUCTS, ADXOPT2_SETS, MIXED_IP,
                              BINSEARCH_QIP_EXACT, NOISYBINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                              NOISYBINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_EXACT]
    },
    'constrained_small'       : {
        'price_range'                        : [1000],
        'num_prods'                          : small_array,
        'repeat_count'                       : RC,
        'prob_v0'                            : None,
        # 'max_assortment_size': 5,
        'max_assortment_size_num_prods_ratio': 0.3,
        'algorithm_list'                     : [ADXOPT1_PRODUCTS_CONSTRAINED, ADXOPT2_SETS_CONSTRAINED,
                                                MIXED_IP_CONSTRAINED,
                                                REVENUE_ORDERED_CONSTRAINED, BINSEARCH_QIP_EXACT_CONSTRAINED,
                                                BINSEARCH_QIP_MTHREAD_CONSTRAINED,
                                                NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    'constrained_tafeng_small': {
        'num_prods'          : small_array,
        'repeat_count'       : RC,
        'prob_v0'            : None,
        'max_assortment_size': 5,
        # 'max_assortment_size_num_prods_ratio': 0.3,
        'parent_model_file'  : 'synthetic_models/models/tafeng.pkl',
        # 'algorithm_list'     : [ADXOPT1_PRODUCTS_CONSTRAINED, ADXOPT2_SETS_CONSTRAINED,
        #                         MIXED_IP_CONSTRAINED,
        #                         REVENUE_ORDERED_CONSTRAINED, BINSEARCH_QIP_EXACT_CONSTRAINED,
        #                         BINSEARCH_QIP_MTHREAD_CONSTRAINED,
        #                         NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED]
        'algorithm_list'   : [REVENUE_ORDERED_CONSTRAINED]
    },
    'constrained_uci_small'   : {
        'num_prods'          : small_array,
        'repeat_count'       : RC,
        'prob_v0'            : None,
        'max_assortment_size': 5,
        # 'max_assortment_size_num_prods_ratio': 0.3,
        'parent_model_file'  : 'synthetic_models/models/uci.pkl',
        'algorithm_list'     : [ADXOPT1_PRODUCTS_CONSTRAINED, ADXOPT2_SETS_CONSTRAINED,
                                MIXED_IP_CONSTRAINED,
                                REVENUE_ORDERED_CONSTRAINED, BINSEARCH_QIP_EXACT_CONSTRAINED,
                                BINSEARCH_QIP_MTHREAD_CONSTRAINED,
                                NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED]
    }
})

mnl_set_dict = {
    'mnl_revenue_ordered_uci'       : {
        'price_range'      : [1000],
        'num_prods'        : small_array + medium_array + large_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'parent_model_file': 'synthetic_models/models/uci_mnl.pkl',
        'is_mnl_model'     : True,
        'gt_model'         : 'rcm',
        'algorithm_list'   : [MNL_REVENUE_ORDERED]
    },
    'mnl_uci_gt_rcm_small_medium'   : {
        'price_range'      : [1000],
        'num_prods'        : small_array + medium_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list'   : [BINSEARCHIMPROVED_QIP_EXACT]
    },
    'mnl_uci_gt_rcm_large'          : {
        'price_range'      : [1000],
        'num_prods'        : large_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list'   : [BINSEARCHIMPROVED_QIP_MTHREAD, NOISYBINSEARCHIMPROVED_QIP_MTHREAD]
    },
    'mnl_revenue_ordered_tafeng'    : {
        'price_range'      : [1000],
        'num_prods'        : small_array + medium_array + large_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'parent_model_file': 'synthetic_models/models/tafeng_mnl.pkl',
        'is_mnl_model'     : True,
        'gt_model'         : 'rcm',
        'algorithm_list'   : [MNL_REVENUE_ORDERED]
    },
    'mnl_tafeng_gt_rcm_small_medium': {
        'price_range'      : [1000],
        'num_prods'        : small_array + medium_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list'   : [BINSEARCHIMPROVED_QIP_EXACT]
    },
    'mnl_tafeng_gt_rcm_large'       : {
        'price_range'      : [1000],
        'num_prods'        : large_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list'   : [BINSEARCHIMPROVED_QIP_MTHREAD, NOISYBINSEARCHIMPROVED_QIP_MTHREAD]
    }
}

tcm_set_dict = {
    'tcm_bonmin_tafeng'      : {
        'num_prods'        : small_array,
        'repeat_count'     : RC,
        'parent_model_file': 'synthetic_models/models/tafeng_tcm.pkl',
        'is_tcm_model'     : True,
        'prob_v0'          : 0.3,
        'algorithm_list'   : [BINSEARCH_TCM_BONMIN_EXACT, TCM_BONMIN_MNLIP]
    },
    'tafeng_small_gt_tcm'    : {
        'num_prods'        : small_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'gt_model'         : 'tcm',
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list'   : [BINSEARCHIMPROVED_QIP_EXACT]
    },
    'tafeng_small_gt_tcm_mnl': {
        'num_prods'        : small_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'parent_model_file': 'synthetic_models/models/tafeng_mnl.pkl',
        'is_mnl_model'     : True,
        'gt_model'         : 'tcm',
        'algorithm_list'   : [MNL_REVENUE_ORDERED]
    },
    'tcm_bonmin_uci'         : {
        'num_prods'        : small_array,
        'repeat_count'     : RC,
        'parent_model_file': 'synthetic_models/models/uci_tcm.pkl',
        'is_tcm_model'     : True,
        'prob_v0'          : 0.3,
        'algorithm_list'   : [BINSEARCH_TCM_BONMIN_EXACT, TCM_BONMIN_MNLIP]
    },
    'uci_small_gt_tcm'       : {
        'num_prods'        : small_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'gt_model'         : 'tcm',
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list'   : [BINSEARCHIMPROVED_QIP_EXACT]
    },
    'uci_small_gt_tcm_mnl'   : {
        'num_prods'        : small_array,
        'repeat_count'     : RC,
        'prob_v0'          : 0.3,
        'parent_model_file': 'synthetic_models/models/uci_mnl.pkl',
        'is_mnl_model'     : True,
        'gt_model'         : 'tcm',
        'algorithm_list'   : [MNL_REVENUE_ORDERED]
    }
}

experiment_set_dict.update(tcm_set_dict)

medium_set_dict = {
    # Medium Experiments Config
    'medium'                   : {
        'price_range'   : [1000],
        'num_prods'     : medium_array,
        'repeat_count'  : RC,
        'algorithm_list': [BINSEARCH_QIP_EXACT, NOISYBINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                           NOISYBINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_EXACT]
    },
    'tafeng_medium'            : {
        'num_prods'        : medium_array,
        'repeat_count'     : RC,
        'prob_v0'          : None,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        # 'algorithm_list'   : [BINSEARCH_QIP_EXACT, NOISYBINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
        #                       NOISYBINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_EXACT]
        'algorithm_list'   : [REVENUE_ORDERED]
    },
    'uci_medium'               : {
        'num_prods'        : medium_array,
        'repeat_count'     : RC,
        'prob_v0'          : None,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list'   : [BINSEARCH_QIP_EXACT, NOISYBINSEARCH_QIP_MTHREAD, REVENUE_ORDERED,
                              NOISYBINSEARCHIMPROVED_QIP_MTHREAD, BINSEARCHIMPROVED_QIP_EXACT]
    },
    'constrained_medium'       : {
        'price_range'        : [1000],
        'num_prods'          : medium_array,
        'repeat_count'       : RC,
        'prob_v0'            : None,
        'max_assortment_size': 20,
        # 'max_assortment_size_num_prods_ratio': 0.3,
        'algorithm_list'     : [REVENUE_ORDERED_CONSTRAINED,
                                BINSEARCH_QIP_MTHREAD_CONSTRAINED,
                                NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED]
        # , BINSEARCH_QIP_EXACT_CONSTRAINED]
    },
    'constrained_tafeng_medium': {
        'num_prods'          : medium_array,
        'repeat_count'       : RC,
        'prob_v0'            : None,
        'max_assortment_size': 20,
        # 'max_assortment_size_num_prods_ratio': 0.3,
        'parent_model_file'  : 'synthetic_models/models/tafeng.pkl',
        # 'algorithm_list'     : [REVENUE_ORDERED_CONSTRAINED,
        #                         BINSEARCH_QIP_MTHREAD_CONSTRAINED,
        #                         NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED,
        #                         BINSEARCH_QIP_EXACT_CONSTRAINED]
        'algorithm_list'   : [REVENUE_ORDERED_CONSTRAINED]
    },
    'constrained_uci_medium'   : {
        'num_prods'          : medium_array,
        'repeat_count'       : RC,
        'prob_v0'            : None,
        'max_assortment_size': 20,
        # 'max_assortment_size_num_prods_ratio': 0.3,
        'parent_model_file'  : 'synthetic_models/models/uci.pkl',
        'algorithm_list'     : [REVENUE_ORDERED_CONSTRAINED,
                                BINSEARCH_QIP_MTHREAD_CONSTRAINED,
                                NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED,
                                BINSEARCH_QIP_EXACT_CONSTRAINED]
    }
}

experiment_set_dict.update(medium_set_dict)

v0_levels = range(10, 100, 20)
v0_configs = {
    f'v0_{xr}_synthetic': {
        'price_range'   : [1000],
        'num_prods'     : large_array,
        'repeat_count'  : RC,
        'prob_v0'       : xr / 100,
        'algorithm_list': [NOISYBINSEARCHIMPROVED_QIP_MTHREAD, NOISYBINSEARCH_QIP_MTHREAD]
    }
    for xr in v0_levels
}

experiment_set_dict.update(v0_configs)

# v00_levels = range(1, 20, 4)
# v00_configs = {
#     f'v00_{xr}_synthetic': {
#         'price_range': [1000],
#         'num_prods': large_array,
#         'repeat_count': RC,
#         'prob_v0': xr / 100,
#         'algorithm_list': [BINSEARCH_QIP_MTHREAD,BINSEARCHIMPROVED_QIP_MTHREAD, NOISYBINSEARCHIMPROVED_QIP_MTHREAD,
#                            NOISYBINSEARCH_QIP_MTHREAD]
#     }
#     for xr in v00_levels
# }
#
# experiment_set_dict.update(v00_configs)

large_set_dict = {
    # Large Experiments Config
    'large'                   : {
        'price_range'   : [1000],
        'num_prods'     : large_array,
        'repeat_count'  : RC,
        'algorithm_list': [NOISYBINSEARCH_QIP_MTHREAD, NOISYBINSEARCHIMPROVED_QIP_MTHREAD,
                           BINSEARCHIMPROVED_QIP_EXACT, REVENUE_ORDERED]
    },
    'constrained_large'       : {
        'price_range'                        : [1000],
        'num_prods'                          : large_array,
        'repeat_count'                       : RC,
        'prob_v0'                            : None,
        # 'max_assortment_size': 20,
        'max_assortment_size_num_prods_ratio': 0.3,
        'algorithm_list'                     : [REVENUE_ORDERED_CONSTRAINED, BINSEARCH_QIP_MTHREAD_CONSTRAINED,
                                                NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    'tafeng_large'            : {
        'num_prods'        : large_array,
        'repeat_count'     : RC,
        'prob_v0'          : None,
        'parent_model_file': 'synthetic_models/models/tafeng.pkl',
        'algorithm_list'   : [NOISYBINSEARCH_QIP_MTHREAD, NOISYBINSEARCHIMPROVED_QIP_MTHREAD,
                              BINSEARCHIMPROVED_QIP_EXACT, REVENUE_ORDERED]
    },
    'uci_large'               : {
        'num_prods'        : large_array,
        'repeat_count'     : RC,
        'prob_v0'          : None,
        'parent_model_file': 'synthetic_models/models/uci.pkl',
        'algorithm_list'   : [NOISYBINSEARCH_QIP_MTHREAD, NOISYBINSEARCHIMPROVED_QIP_MTHREAD,
                              BINSEARCHIMPROVED_QIP_EXACT, REVENUE_ORDERED]
    },
    'constrained_tafeng_large': {
        'num_prods'                          : large_array,
        'repeat_count'                       : RC,
        'prob_v0'                            : None,
        # 'max_assortment_size': 20,
        'max_assortment_size_num_prods_ratio': 0.3,
        'parent_model_file'                  : 'synthetic_models/models/tafeng.pkl',
        'algorithm_list'                     : [REVENUE_ORDERED_CONSTRAINED, BINSEARCH_QIP_MTHREAD_CONSTRAINED,
                                                NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED]
    },
    'constrained_uci_large'   : {
        'num_prods'                          : large_array,
        'repeat_count'                       : RC,
        'prob_v0'                            : None,
        # 'max_assortment_size': 20,
        'max_assortment_size_num_prods_ratio': 0.3,
        'parent_model_file'                  : 'synthetic_models/models/uci.pkl',
        'algorithm_list'                     : [REVENUE_ORDERED_CONSTRAINED, BINSEARCH_QIP_MTHREAD_CONSTRAINED,
                                                NOISYBINSEARCH_QIP_MTHREAD_CONSTRAINED]
    }
}

experiment_set_dict.update(large_set_dict)

experiment_set_dict.update(mnl_set_dict)


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

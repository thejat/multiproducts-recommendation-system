from algorithms_config import *

default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}

ucm_model_dir = f'results/ucm_optim/%s_experiments/ucm_models'
ucm_solution_dir = 'results/ucm_optim/%s_experiments/ucm_solutions'
ucm_summary_dir = 'results/ucm_optim/%s_experiments'


experiment_set_dict = {
#Small Experiments Config
    'small':{
        'price_range':[100, 1000, 10000],
        'num_prods': list(range(5, 6, 5)),
        'repeat_count': 1,
        'Hset_multiplier_list':[0, 0.25, 0.5, 1],
        'algorithm_list': [MIXED_IP,ADXOPT1_PRODUCTS, ADXOPT2_SETS]
    },
#Medium Experiments Config
    'medium':{
        'price_range':[100, 1000, 10000],
        'num_prods': list(range(100, 500, 50)),
        'repeat_count':20,
        'Hset_multiplier_list':[0, 0.25, 0.5, 1],
        'algorithm_list': [MIXED_IP, ADXOPT2_SETS, ADXOPT1_PRODUCTS]
    },
}

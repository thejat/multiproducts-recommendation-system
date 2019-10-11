from algorithms_config import *

default_meta = {'eps': 1e-3, 'print_results': True, 'print_debug': True}
price_range_list = [100, 1000, 10000]

rcm_model_dir = f'results/rcm_optim/%s_experiments/rcm_models'
rcm_solution_dir = 'results/rcm_optim/%s_experiments/rcm_solutions'
rcm_summary_dir = 'results/rcm_optim/%s_experiments'


experiment_set_dict = {
#Small Experiments Config
    'small':{
        'price_range':[100, 1000, 10000],
        'num_prods': list(range(5, 25, 5)),
        'repeat_count': 2,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, MIXED_IP,
                      ADXOPT1_PRODUCTS, ADXOPT2_SETS]
    },
#Medium Experiments Config
    'medium':{
        'price_range':[100, 1000, 10000],
        'num_prods': list(range(100, 500, 50)),
        'repeat_count':20,
        'algorithm_list': [BINSEARCH_QIP_EXACT, BINSEARCH_QIP_MTHREAD, ADXOPT2_SETS, ADXOPT1_PRODUCTS]
    },
# Large Experiments Config
    'large':{
        'price_range':[100, 1000, 10000],
        'num_prods': list(range(500, 5000, 500)),
        'repeat_count': 20,
        'algorithm_list':[BINSEARCH_QIP_MTHREAD]
    }
}

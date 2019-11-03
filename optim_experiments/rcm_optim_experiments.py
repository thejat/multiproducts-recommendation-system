import pickle
import os
from datetime import datetime
from synthetic_models.rcm_synthetic_model import generate_two_restricted_choice_model, generate_derived_rcm_choice_model
from optimization.rcm_optim import run_rcm_optimization
from copy import deepcopy
from threading import Thread, Lock
import time
from queue import Queue
from optimization.rcm_optim import compare_nn_preprocess
import traceback
import logging

logger = logging.getLogger(__name__)


def run_rcm_experiments_v2(model_dir, algorithm_list, meta_default, price_range_list, parent_model_file,
                           prod_count_list, repeat_count,
                           output_dir='tmp/solutions/rcm_models/v2', prob_v0=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    experiment_summary = []
    price_ranges = price_range_list
    if price_range_list is None:
        price_ranges = [parent_model_file.split('.')[0].split("/")[-1]]
    logger.info(f"OUTPUT Directory for results: {output_dir}")
    for price_range in price_ranges:
        for num_prods in prod_count_list:
            for repeat_id in range(repeat_count):
                model_filepath = f'{model_dir}/rcm_model_{price_range}_{num_prods}_{repeat_id}.pkl'
                if not os.path.exists(model_filepath):
                    logger.warning(f"No RCM model File {model_filepath} found, skipping...")
                    continue
                with open(model_filepath, 'rb') as f:
                    model_dict = pickle.load(f)
                rcm_model = model_dict['rcm_model']
                if prob_v0 is not None:
                    logger.info(f"Prob v0 is {prob_v0} not None, Updating current RCM Model...")
                    v_sum = sum(rcm_model['v']) + sum(rcm_model['v2'].values()) - rcm_model['v'][0]
                    rcm_model['v'][0] = (prob_v0 * v_sum) / (1 - prob_v0)
                    meta_default['prob_v0'] = prob_v0
                for optim_algo_dict in algorithm_list:
                    try:
                        meta = deepcopy(meta_default)
                        meta.update(optim_algo_dict)
                        if 'comparison_function' in meta.keys():
                            if 'nn' in meta['comparison_function']:
                                index_filepath = \
                                    f'{model_dir}/nn_index_cache/rcm_nn_index_{price_range}_{num_prods}_{repeat_id}.pkl'
                                meta.update({'index_filepath': index_filepath})

                        if prob_v0 is None:
                            model_solve_filepath = \
                                f'{output_dir}/rcm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{repeat_id}.pkl'
                        else:
                            model_solve_filepath = \
                                f'{output_dir}/rcm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{repeat_id}_v0_{int(prob_v0 * 100)}.pkl'
                        if not os.path.exists(model_solve_filepath):
                            sol_dict = {key: model_dict[key] for key in model_dict.keys() if not (key == 'rcm_model')}
                            sol_dict.update(meta)
                            rcm_solution = run_rcm_optimization(meta['algo'], num_prods, num_prods, rcm_model, meta)
                            sol_dict.update(rcm_solution)
                            with open(model_solve_filepath, 'wb') as f:
                                sol_dict.update(model_dict)
                                pickle.dump(sol_dict, f)
                                logger.info(f"Optimized RCM Model {model_solve_filepath.split('/')[-1]}...\n")
                        else:
                            with open(model_solve_filepath, 'rb') as f:
                                sol_dict = pickle.load(f)
                            logger.info(f"Retrieved RCM Solution {model_solve_filepath.split('/')[-1]}...\n")
                        if 'rcm_model' in sol_dict.keys():
                            del sol_dict['rcm_model']
                        experiment_summary.append(sol_dict)
                    except Exception as e:
                        logger.error(f"\n\nFailed For {model_filepath}")
                        logger.error(f"Algorithm Details")
                        logger.error(optim_algo_dict)
                        logger.error(f"Exception Details:{str(e)}\n\n")
                        traceback.print_exc()
    return experiment_summary


def dump_rcm_models(price_range_list, prod_count_list, repeat_count, dump_dir='tmp/rcm_models/v2/', prob_v0=None):
    # Check If Dump Dir Exists
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    for price_range in price_range_list:
        for num_prod in prod_count_list:
            for i in range(repeat_count):
                dump_filename = f'rcm_model_{price_range}_{num_prod}_{i}.pkl'
                if not os.path.exists(f'{dump_dir}/{dump_filename}'):
                    model_dict = {'price_range': price_range, 'num_prod': num_prod, 'repeat_id': i,
                                  'time_of_creation': time_now}
                    rcm_model = generate_two_restricted_choice_model(price_range, num_prod, prob_v0=prob_v0)
                    model_dict.update({'rcm_model': rcm_model})
                    with open(f'{dump_dir}/{dump_filename}', 'wb') as f:
                        pickle.dump(model_dict, f)
                        logger.info(
                            f"Created RCM Model for price range: {price_range}, num products:{num_prod}, repeat_id:{i}")

    return None


def dump_derived_rcm_models(model_filepath, prod_count_list, repeat_count, dump_dir='tmp/rcm_models/v2/', prob_v0=None,
                            is_mnl=False):
    # Check If Dump Dir Exists
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    with open(model_filepath, 'rb') as f:
        parent_rcm_model = pickle.load(f)
        parent_modelname = model_filepath.split(".")[0].split("/")[-1]

    for num_prod in prod_count_list:
        for i in range(repeat_count):
            dump_filename = f'rcm_model_{parent_modelname}_{num_prod}_{i}.pkl'
            if not os.path.exists(f'{dump_dir}/{dump_filename}'):
                model_dict = {'parent_model': parent_modelname, 'num_prod': num_prod, 'repeat_id': i,
                              'time_of_creation': time_now}
                if is_mnl:
                    rcm_filename = f'rcm_model_{parent_modelname.split("_mnl")[0]}_{num_prod}_{i}.pkl'
                    rcm_filepath = f'{dump_dir}/{rcm_filename}'
                    mnl_v0, selected_products = get_selected_rcm_model_products(rcm_filepath)
                    rcm_model = generate_derived_rcm_choice_model(parent_rcm_model, num_prod, prob_v0=prob_v0,
                                                                  is_mnl=is_mnl, selected_products=selected_products,
                                                                  mnl_v0=mnl_v0)
                else:
                    rcm_model = generate_derived_rcm_choice_model(parent_rcm_model, num_prod, prob_v0=prob_v0)
                model_dict.update({'rcm_model': rcm_model})
                # dump_filename = f'rcm_model_{parent_modelname}_{num_prod}_{i}.pkl'
                with open(f'{dump_dir}/{dump_filename}', 'wb') as f:
                    pickle.dump(model_dict, f)
                    logger.info(
                        f"Created RCM Model for parent model: {parent_modelname}, num products:{num_prod}, repeat_id:{i}")

    return None

def get_selected_rcm_model_products(rcm_filepath):
    if os.path.exists(rcm_filepath):
        model_dict = pickle.load(open(rcm_filepath,'rb'))
        return model_dict['rcm_model']['v'][0], model_dict['rcm_model']['product_ids']
    else:
        logger.error(f"Corresponding RCM Model {rcm_filepath} is not available, Selected Products Set to None...")
        return None

def cache_nn_index_mthread(price_range_list, prod_count_list, repeat_count, model_dir='tmp/rcm_models/v2/',
                           num_threads=10):
    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    index_cache_dir = f'{model_dir}/nn_index_cache'
    if not os.path.exists(index_cache_dir):
        os.makedirs(index_cache_dir)

    message_queue = Queue()
    threadlist = []
    for i in range(num_threads):
        worker = Thread(target=cache_nn_run_subroutine,
                        args=(message_queue, i, model_dir, index_cache_dir))
        worker.setDaemon(True)
        threadlist.append(worker)
        worker.start()

    for price_range in price_range_list:
        for num_prod in prod_count_list:
            for repeat_id in range(repeat_count):
                message_queue.put((price_range, num_prod, repeat_id))
    logger.info('*** Main thread waiting....***')
    message_queue.join()
    logger.info('*** Queue Messages Done... ***')
    # Push end loop message for threads
    for i in range(num_threads):
        message_queue.put(None)

    for t in threadlist:
        t.join()
    logger.info("****Worker Threads Done")
    return None


def cache_nn_run_subroutine(message_queue, threadID, model_dir, index_cache_dir):
    logger.info(f"***Starting Thread:{threadID}***")
    while True:
        queue_message = message_queue.get()
        if queue_message is None:
            break
        price_range, num_prod, repeat_id = queue_message
        index_filename = f'rcm_nn_index_{price_range}_{num_prod}_{repeat_id}.pkl'
        if not os.path.exists(f'{index_cache_dir}/{index_filename}'):
            model_filename = f'rcm_model_{price_range}_{num_prod}_{repeat_id}.pkl'
            with open(f'{model_dir}/{model_filename}', 'rb') as f:
                model_dict = pickle.load(f)
            prices = model_dict['rcm_model']['p']
            index_dict = {'approx_index': {}, 'exact_index': {}}
            index_dict['approx_index']['db'], _, index_dict['approx_index']['normConst'], index_dict['approx_index'][
                'ptsTemp'] = compare_nn_preprocess(num_prod, min(num_prod, 100), prices, 'nn-approx')
            index_dict['exact_index']['db'], _, index_dict['exact_index']['normConst'], index_dict['exact_index'][
                'ptsTemp'] = compare_nn_preprocess(num_prod, min(num_prod, 100), prices, 'nn-exact')
            with open(f'{index_cache_dir}/{index_filename}', 'wb') as f:
                pickle.dump(index_dict, f)
            logger.info(
                f"Created index dict for threadID:{threadID}, price range: {price_range}, num products:{num_prod}, repeat_id:{repeat_id}")
        else:
            logger.info(
                f"Already Exists: index dict threadID:{threadID}, price: {price_range}, num products:{num_prod}, repeat_id:{repeat_id}")
        message_queue.task_done()

    return None

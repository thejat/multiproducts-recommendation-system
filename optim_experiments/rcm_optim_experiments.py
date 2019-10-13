import pickle
import os
from datetime import datetime
from synthetic_models.rcm_synthetic_model import generate_two_restricted_choice_model
from optimization.rcm_optim import run_rcm_optimization
from copy import deepcopy
from threading import Thread, Lock
import time
from queue import Queue
from optimization.rcm_optim import compare_nn_preprocess
import traceback


def run_rcm_experiments_v2(model_dir, algorithm_list, meta_default, price_range_list, prod_count_list, repeat_count,
                           output_dir='tmp/solutions/rcm_models/v2'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    experiment_summary = []

    for price_range in price_range_list:
        for num_prods in prod_count_list:
            for repeat_id in range(repeat_count):
                model_filepath = f'{model_dir}/rcm_model_{price_range}_{num_prods}_{repeat_id}.pkl'
                if not os.path.exists(model_filepath):
                    print(f"No RCM model File {model_filepath} found, skipping...")
                    continue
                with open(model_filepath, 'rb') as f:
                    model_dict = pickle.load(f)
                rcm_model = model_dict['rcm_model']
                for optim_algo_dict in algorithm_list:
                    try:
                        meta = deepcopy(meta_default)
                        meta.update(optim_algo_dict)
                        if 'comparison_function' in meta.keys():
                            if 'nn' in meta['comparison_function']:
                                index_filepath = \
                                    f'{model_dir}/nn_index_cache/rcm_nn_index_{price_range}_{num_prods}_{repeat_id}.pkl'
                                meta.update({'index_filepath': index_filepath})
                        model_solve_filepath = \
                            f'{output_dir}/rcm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{repeat_id}.pkl'
                        if not os.path.exists(model_solve_filepath):
                            sol_dict = deepcopy(model_dict)
                            sol_dict.update(meta)
                            rcm_solution = run_rcm_optimization(meta['algo'], num_prods, num_prods, rcm_model, meta)
                            sol_dict.update(rcm_solution)
                            with open(model_solve_filepath, 'wb') as f:
                                pickle.dump(sol_dict, f)
                                print(f"Optimized RCM Model {model_solve_filepath}...")
                        else:
                            with open(model_solve_filepath, 'rb') as f:
                                sol_dict = pickle.load(f)
                            print(f"Retrieved RCM Solution {model_solve_filepath}...")
                        del sol_dict['rcm_model']
                        experiment_summary.append(sol_dict)
                    except Exception as e:
                        print(f"\n\nFailed For {model_filepath}")
                        print(f"Algorithm Details")
                        print(optim_algo_dict)
                        print(f"Exception Details:{str(e)}\n\n")
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
                model_dict = {'price_range': price_range, 'num_prod': num_prod, 'repeat_id': i,
                              'time_of_creation': time_now}
                rcm_model = generate_two_restricted_choice_model(price_range, num_prod,prob_v0=prob_v0)
                model_dict.update({'rcm_model': rcm_model})
                dump_filename = f'rcm_model_{price_range}_{num_prod}_{i}.pkl'
                with open(f'{dump_dir}/{dump_filename}', 'wb') as f:
                    pickle.dump(model_dict, f)
                    print(f"Created RCM Model for price range: {price_range}, num products:{num_prod}, repeat_id:{i}")

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
    print('*** Main thread waiting....***')
    message_queue.join()
    print('*** Queue Messages Done... ***')
    # Push end loop message for threads
    for i in range(num_threads):
        message_queue.put(None)

    for t in threadlist:
        t.join()
    print("****Worker Threads Done")
    return None


def cache_nn_run_subroutine(message_queue, threadID, model_dir, index_cache_dir):
    print(f"***Starting Thread:{threadID}***")
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
            print(
                f"Created index dict for threadID:{threadID}, price range: {price_range}, num products:{num_prod}, repeat_id:{repeat_id}")
        else:
            print(
                f"Already Exists: index dict threadID:{threadID}, price: {price_range}, num products:{num_prod}, repeat_id:{repeat_id}")
        message_queue.task_done()


    return None

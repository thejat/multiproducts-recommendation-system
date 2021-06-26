import pickle
import os
from datetime import datetime
from synthetic_models.ucm_synthetic_model import generate_universal_choice_model, generate_derived_ucm_choice_model
from optimization.ucm_optim import run_ucm_optimization
from copy import deepcopy
import time


def run_ucm_experiments_v2(model_dir, algorithm_list, meta_default, price_range_list, parent_model_file, prod_count_list, repeat_count,
                           Hset_multiplier_list,
                           output_dir='tmp/solutions/ucm_models/v2'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    experiment_summary = []
    price_ranges = price_range_list
    Hset_multipliers = Hset_multiplier_list
    if price_range_list is None:
        price_ranges = [parent_model_file.split('.')[0].split("/")[-1]]
        Hset_multipliers=[-1]


    for price_range in price_ranges:
        for num_prods in prod_count_list:
            for Hset_multiplier in Hset_multipliers:
                n_Hset_count = int(num_prods * (num_prods - 1) * Hset_multiplier / 2)
                for repeat_id in range(repeat_count):
                    if price_range_list is None:
                        model_filepath = f'{model_dir}/ucm_model_{price_range}_{num_prods}_{repeat_id}.pkl'
                    else:
                        model_filepath = f'{model_dir}/ucm_model_{price_range}_{num_prods}_{n_Hset_count}_{repeat_id}.pkl'
                    if not os.path.exists(model_filepath):
                        print(f"No UCM model File {model_filepath} found, skipping...")
                        continue
                    with open(model_filepath, 'rb') as f:
                        model_dict = pickle.load(f)
                    ucm_model = model_dict['ucm_model']
                    for optim_algo_dict in algorithm_list:
                        try:
                            meta = deepcopy(meta_default)
                            meta.update(optim_algo_dict)
                            if price_range_list is None:
                                model_solve_filepath = \
                                    f'{output_dir}/ucm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{repeat_id}.pkl'
                            else:
                                model_solve_filepath = \
                                    f'{output_dir}/ucm_model_{meta["solution_id"]}_{price_range}_{num_prods}_{n_Hset_count}_{repeat_id}.pkl'

                            if not os.path.exists(model_solve_filepath):
                                sol_dict = deepcopy(model_dict)
                                sol_dict.update(meta)
                                ucm_solution = run_ucm_optimization(meta['algo'], num_prods, num_prods, ucm_model, meta)
                                sol_dict.update(ucm_solution)
                                with open(model_solve_filepath, 'wb') as f:
                                    pickle.dump(sol_dict, f)
                                    print(f"Optimized UCM Model {model_solve_filepath}...")
                            else:
                                with open(model_solve_filepath, 'rb') as f:
                                    sol_dict = pickle.load(f)
                                print(f"Retrieved UCM Solution {model_solve_filepath}...")
                            del sol_dict['ucm_model']
                            experiment_summary.append(sol_dict)
                        except Exception as e:
                            print(f"\n\nFailed For {model_filepath}")
                            print(f"Algorithm Details")
                            print(optim_algo_dict)
                            print(f"Exception Details:{str(e)}\n\n")
    return experiment_summary


def dump_ucm_models(price_range_list, prod_count_list, Hset_multiplier_list, repeat_count,
                    dump_dir='tmp/ucm_models/v2/'):
    # Check If Dump Dir Exists
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    for price_range in price_range_list:
        for num_prod in prod_count_list:
            for Hset_multiplier in Hset_multiplier_list:
                n_Hset_count = int(num_prod * (num_prod - 1) * Hset_multiplier / 2)
                for i in range(repeat_count):
                    dump_filename = f'ucm_model_{price_range}_{num_prod}_{n_Hset_count}_{i}.pkl'
                    if not os.path.exists(f'{dump_dir}/{dump_filename}'):
                        model_dict = {'price_range': price_range, 'num_prod': num_prod,
                                      'nHset_count': n_Hset_count, 'repeat_id': i,
                                      'time_of_creation': time_now}
                        ucm_model = generate_universal_choice_model(price_range, num_prod, n_Hset_count)
                        model_dict.update({'ucm_model': ucm_model})
                        with open(f'{dump_dir}/{dump_filename}', 'wb') as f:
                            pickle.dump(model_dict, f)
                            print(
                                f"Created UCM Model for price range: {price_range}, num products:{num_prod}, "
                                f"Hset_count:{n_Hset_count} repeat_id:{i}")

    return None


def dump_derived_ucm_models(model_filepath, prod_count_list, repeat_count, dump_dir='tmp/ucm_models/v2/'):
    # Check If Dump Dir Exists
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    with open(model_filepath, 'rb') as f:
        parent_ucm_model = pickle.load(f)
        parent_modelname = model_filepath.split(".")[0].split("/")[-1]

    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")

    for num_prod in prod_count_list:
        for i in range(repeat_count):
            dump_filename = f'ucm_model_{parent_modelname}_{num_prod}_{i}.pkl'
            if not os.path.exists(f'{dump_dir}/{dump_filename}'):
                model_dict = {'parent_model': parent_modelname, 'num_prod': num_prod,
                              'repeat_id': i,
                              'time_of_creation': time_now}
                ucm_model = generate_derived_ucm_choice_model(parent_ucm_model, num_prod)
                model_dict.update({'ucm_model': ucm_model})
                with open(f'{dump_dir}/{dump_filename}', 'wb') as f:
                    pickle.dump(model_dict, f)
                    print(
                        f"Created UCM Model for parent model: {parent_modelname}, num products:{num_prod}, "
                        f"repeat_id:{i}")

    return None

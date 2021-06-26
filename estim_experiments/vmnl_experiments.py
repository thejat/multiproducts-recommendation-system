import numpy as np
import pandas as pd
from model_learning import vmnl_model as vmnl
import time


def run_vmnl_experiment(traindatafiles, testdatafiles, results_dir):
    likelihoods = {}
    for traindatafile, testdatafile in zip(traindatafiles, testdatafiles):
        print(f'running experiment for {traindatafile}...')
        train_data = vmnl.read_data(traindatafile)
        test_data = vmnl.read_data(testdatafile)
        start_time = time.time()
        model = vmnl.initialize_model(train_data)
        # vmnl.log_likelihood(model, train_data)
        vmnl.learn_utilities(model, train_data)
        training_time = time.time() - start_time
        print(f'finished training for VMNL model in {training_time} secs')
        num_parameters = len(list(model.utilities.keys()))
        train_likelihood = vmnl.log_likelihood(model, train_data)
        start_time = time.time()
        test_likelihood = vmnl.log_likelihood(model, test_data)
        testing_time = time.time() - start_time

        likelihoods[f'{traindatafile.split("train_data_")[-1]}'] = [num_parameters, train_likelihood,
                                                                    test_likelihood]
        print(
            f'{traindatafile.split("train")}:: #parameters: {num_parameters},training_time: {training_time}, testing_time: {testing_time}'
            f'train_likelihood: {train_likelihood}, test_likelihoods: {test_likelihood}')
    df_results = pd.DataFrame.from_dict(likelihoods, orient='index')
    df_results.columns = ['num_params', 'train_likelihood', 'test_likelihood']
    df_results.to_csv(f'{results_dir}/vmnl_results.csv')

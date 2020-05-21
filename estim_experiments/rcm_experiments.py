import numpy as np
import pandas as pd
from model_learning import rcm_model as rcm
from model_learning import ucm_model as ucm
import time

def run_rcm_experiment(traindatafiles, testdatafiles, results_dir):
    likelihoods = {}
    for traindatafile, testdatafile in zip(traindatafiles, testdatafiles):
        print(f'running experiment for {traindatafile}...')
        train_data = rcm.read_data(traindatafile)
        test_data = rcm.read_data(testdatafile)
        start_time = time.time()
        model = rcm.initialize_model(train_data)
        training_time = time.time() - start_time
        print(f'finished training for MNL model in {training_time} secs')
        num_parameters = len(list(model.probs.keys()))
        train_likelihood = rcm.log_likelihood(model, train_data)
        start_time = time.time()
        test_likelihood = rcm.log_likelihood(model, test_data)
        testing_time = time.time() - start_time

        likelihoods[f'{traindatafile.split("train_data_")[-1]}'] = [train_likelihood,
                                                                    test_likelihood]
        print(f'{traindatafile.split("train")}:: #parameters: {num_parameters},training_time: {training_time}, testing_time: {testing_time}'
              f'train_likelihood: {train_likelihood}, test_likelihoods: {test_likelihood}')
    df_results = pd.DataFrame.from_dict(likelihoods, orient='index')
    df_results.columns = ['train_likelihood', 'test_likelihood']
    df_results.to_csv(f'{results_dir}/rcm_results.csv')

#
# univ_datasets = ['bakery.txt', 'instacart-5-25.txt', 'kosarak.txt', 'lastfm-genres.txt', 'walmart-depts.txt',
#                  'walmart-items.txt']
# data_dir = '../../data'
# results_dir = '../results/ucm_results'
# traindatafiles = [f'{data_dir}/train_data_{dataset}' for dataset in univ_datasets]
# testdatafiles = [f'{data_dir}/test_data_{dataset}' for dataset in univ_datasets]
#
# run_rcm_experiment(traindatafiles, testdatafiles, results_dir)

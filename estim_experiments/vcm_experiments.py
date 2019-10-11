import numpy as np
import pandas as pd
from model_learning import vcm_model as vcm
from sklearn.model_selection import train_test_split
import time

'''
This function is Frequency heuristics to learn H
'''


def most_freq_choice_tups(data, num):
    counts = [(count, tup) for (tup, count) in vcm.get_subset_counts(data).items() if len(tup) > 1]
    counts.sort(reverse=True)
    if num == -1:
        return [tup for (count, tup) in counts]
    return [tup for (count, tup) in counts[:num]]


def run_vcm_experiment(traindatafile, testdatafile, results_dir):
    train_data = vcm.read_data(traindatafile)
    test_data = vcm.read_data(testdatafile)
    unique_item_count = max([max(choice) for choice in train_data.choices])
    choices_to_add = most_freq_choice_tups(train_data, -1)
    num_corrections = np.array([0, 0.01, 0.05, 0.2, 0.5, 1]) * len(choices_to_add)
    num_corrections = list(map(int, num_corrections))
    print(num_corrections)
    likelihoods = {}

    for num in num_corrections:
        print(num)
        start_time = time.time()
        model = vcm.initialize_model_with_H(train_data, choices_to_add[:num])
        vcm.learn_model(model, train_data)
        training_time = time.time() - start_time
        train_likelihood = vcm.log_likelihood(model, train_data)
        num_parameters = len(model.utilities) + len(model.H) + len(model.z)
        start_time = time.time()
        test_likelihood = vcm.log_likelihood(model, test_data)
        testing_time = time.time() - start_time
        likelihoods[num] = [train_likelihood, test_likelihood]
        print(f"Got {num} num_corrections, likelihoods {likelihoods[num]}")
        print(f'{traindatafile.split("train")}:: #parameters: {num_parameters},training_time: {training_time}, testing_time: {testing_time}'
              f'train_likelihood: {train_likelihood}, test_likelihoods: {test_likelihood}')

    df_results = pd.DataFrame.from_dict(likelihoods, orient='index')
    df_results.to_csv(f'{results_dir}/{traindatafile.split("train")[-1]}')

import numpy as np
import pandas as pd
from model_learning import ucm_model as ucm
from sklearn.model_selection import train_test_split
import time

'''
This function is Frequency heuristics to learn H
'''


def most_freq_choice_tups(data, num):
    counts = [(count, tup) for (tup, count) in ucm.get_subset_counts(data).items() if len(tup) > 1]
    counts.sort(reverse=True)
    if num == -1:
        return [tup for (count, tup) in counts]
    return [tup for (count, tup) in counts[:num]]


'''
This Function return maximum size of H set, i.e total unique choices we can put into H, or 1000,
whichever is minimum
'''


def maximum_H_size(data):
    return min(max([max(choice) for choice in data.choices]), 1000)


'''
This is main function which runs experiments on data, trying to see imporvements in
log likelihood values base on size of correction set H...
'''


def universal_improvements(data, num_updates, basename, update_type, train_size=0.8):
    # split whole data in training and test sizes
    update_log_likelihoods = np.zeros(num_updates + 1)
    training_choices, test_choices, training_sizes, test_sizes = train_test_split(data.choices, data.sizes,
                                                                                  train_size=train_size)

    training_data = ucm.UniversalChoiceDataset(training_sizes, training_choices)
    test_data = ucm.UniversalChoiceDataset(test_sizes, test_choices)

    # initialization step
    model = ucm.initialize_model(training_data)
    update_log_likelihoods[0] = ucm.log_likelihood(model, test_data)

    # get Item counts
    total_items = max([max(choice) for choice in data.choices])
    item_counts = {i: 0 for i in range(1, total_items + 1)}
    for choice in data.choices:
        for item in choice:
            item_counts[item] += 1

    choices_to_add = most_freq_choice_tups(training_data, num_updates)
    if update_type == 'f':
        pass  # Need to keep frequency based only
    elif update_type == 'nl':
        # Normalized Lift Based Updates
        lifts = []
        for (choice_tup, subset_count) in ucm.get_subset_counts(training_data).items():
            if len(choice_tup) > 1:
                subset_item_count = [item_counts[item] for item in choice_tup]
                lifts.append((subset_count ** 2 / np.prod(subset_item_count), choice_tup))
        lifts.sort(reverse=True)
        choices_to_add = [choice_tup for (_, choice_tup) in lifts[:num_updates]]
        # TODO Cross verify nl numbers for training data
    elif update_type == 'l':
        # Lift Based Updates
        lifts = []
        for (choice_tup, subset_count) in ucm.get_subset_counts(training_data).items():
            if len(choice_tup) > 1:
                subset_item_count = [item_counts[item] for item in choice_tup]
                lifts.append((subset_count / np.prod(subset_item_count), choice_tup))
        lifts.sort(reverse=True)
        choices_to_add = [choice_tup for (_, choice_tup) in lifts[:num_updates]]
        # TODO Cross verify lift numbers for training data
    else:
        print(f"Unknown Update Type:{update_type}")

    # Calculate log likehoods with adding cirrection factors
    for (i, choice) in enumerate(choices_to_add):
        ucm.add_to_H(model, choice)
        update_log_likelihoods[i + 1] = ucm.log_likelihood(model, test_data)

    pd.Series(update_log_likelihoods).to_csv(f'results/{basename}_{update_type}.csv', index=True)

    return None


'''
This Function generates:
 1- counts of Negative corrections at each H size step
 2- See which are top 5 biggest correction improvements
'''


def correction_experiments(datapath, datafile, biggest_correction_count=5, model_neg_correction=True,
                           model_biggest_corrections=True):
    data = ucm.read_data(f'{datapath}/{datafile}')
    num_updates = maximum_H_size(data)
    choices_to_add = most_freq_choice_tups(data, num_updates)
    model = ucm.initialize_model(data)

    if model_neg_correction:
        # Model Negative Corrections
        num_negative_corrections = [0]
        for (i, choice) in enumerate(choices_to_add):
            ucm.add_to_H(model, choice)
            count = 0
            for (subset, val) in model.H.items():
                separable_prob = np.prod([model.probs[item] for item in subset])
                gamma = model.gammas[len(subset)]
                correction = val - (gamma * separable_prob)
                if correction < 0:
                    count += 1
            # print(f"Iteration:{i} of {num_updates}:{count} corrections < 0")
            num_negative_corrections.append(count)
        pd.Series(num_negative_corrections).to_csv(f'results/{datafile}-freq-neg-corrections.txt', header=False)

    model = ucm.initialize_model(data)
    if model_biggest_corrections:
        results = []
        for choice in choices_to_add:
            ucm.add_to_H(model, choice)
        for (subset, val) in model.H.items():
            separable_prob = np.prod([model.probs[item] for item in subset])
            gamma = model.gammas[len(subset)]
            correction = val - (gamma * separable_prob)
            results.append(tuple((correction, subset)))

        results.sort()
        # Write Most Biggest corrections
        pd.Series(results[:biggest_correction_count]).to_csv(f'results/{datafile}-biggest-neg-corrections.txt',
                                                             index=False, header=False)
        pd.Series(results[-biggest_correction_count:]).to_csv(f'results/{datafile}-biggest-pos-corrections.txt',
                                                              index=False, header=False)

    return None


'''
This function consolidates all Heuristic experiments to generate models
'''


def universal_likelihood_experiments(datapath, datafile):
    print(f"Running likelihood experiments for {datafile}")
    data = ucm.read_data(f'{datapath}/{datafile}')
    num_updates = maximum_H_size(data)

    print("Running Frequency Heuristics likelihoods..")
    start_time = time.time()
    universal_improvements(data, num_updates, datafile, 'f')
    print("Done likelihood experiments in %.3f secs.." % (time.time() - start_time))

    print("Running Lift Heuristics likelihoods..")
    start_time = time.time()
    universal_improvements(data, num_updates, datafile, 'l')
    print("Done likelihood experiments in %.3f secs.." % (time.time() - start_time))

    print("Running Normalized Lift Heuristics likelihoods..")
    start_time = time.time()
    universal_improvements(data, num_updates, datafile, 'nl')
    print("Done likelihood experiments in %.3f secs.." % (time.time() - start_time))

    return None


def run_ucm_experiment(traindatafile, testdatafile, results_dir):
    train_data = ucm.read_data(traindatafile)
    test_data = ucm.read_data(testdatafile)
    # unique_item_count = max([max(choice) for choice in train_data.choices])
    choices_to_add = most_freq_choice_tups(train_data, -1)
    # num_corrections = np.array([0, 0.01, 0.05, 0.2, 0.5, 1]) * len(choices_to_add)
    num_corrections = np.array([1]) * len(choices_to_add)
    num_corrections = list(map(int, num_corrections))
    print(num_corrections)
    likelihoods = {}

    # model = ucm.initialize_model(train_data)
    # likelihoods[0] = [ucm.log_likelihood(model, train_data), ucm.log_likelihood(model, test_data)]
    # for (i, choice) in enumerate(choices_to_add):
    #     ucm.add_to_H(model, choice)
    #     print(i)
    #     if i in num_corrections:
    #         likelihoods[i] = [ucm.log_likelihood(model, train_data), ucm.log_likelihood(model, test_data)]
    #         print(f"Got {i} num_corrections, likelihoods {likelihoods[i]}")

    for num in num_corrections:
        print(num)
        start_time = time.time()
        model = ucm.initialize_model_with_H(train_data, choices_to_add[:num])
        training_time = time.time() - start_time
        train_likelihood = ucm.log_likelihood(model, train_data)
        num_parameters = len(model.probs)+len(model.H)
        start_time = time.time()
        test_likelihood = ucm.log_likelihood(model, test_data)
        testing_time = time.time()-start_time
        likelihoods[num] = [train_likelihood, test_likelihood]
        print(f"Got {num} num_corrections, likelihoods {likelihoods[num]}")
        print(f'{traindatafile.split("train")}:: #parameters: {num_parameters},training_time: {training_time}, testing_time: {testing_time}'
              f'train_likelihood: {train_likelihood}, test_likelihoods: {test_likelihood}')

    df_results = pd.DataFrame.from_dict(likelihoods, orient='index')
    df_results.to_csv(f'{results_dir}/{traindatafile.split("train")[-1]}')

import numpy as np
from numpy import linalg
import itertools
from scipy.optimize import minimize as Optimizer
from scipy.optimize import fmin_l_bfgs_b as minOptimizer

'''
This Class represents a variable choice dataset in terms of slates, and made choices
'''


class VariableChoiceDataset:
    def __init__(self, slate_sizes, slates, choice_sizes, choices):
        self.slate_sizes = np.array(slate_sizes)
        self.slates = np.array([np.array(slate) for slate in slates])
        self.choice_sizes = np.array(choice_sizes)
        self.choices = np.array([np.array(choice) for choice in choices])


'''
This class represents a variable choice model in terms of Utilities, Z probabilites and correction
factors(H)
'''


class VariableChoiceModel:
    def __init__(self, z, utilities, H):
        self.z = z
        self.utilities = utilities
        self.H = H


'''
This Function reads data and stores it into VariableChoiceDataset
'''

'''
This function gets subsets and their counts for given UniversalChoiceDataset
'''


def get_subset_counts(data):
    subset_counts = {}
    for i in range(len(data.choice_sizes)):
        choice = tuple(data.choices[i])
        if choice not in subset_counts.keys():
            subset_counts[choice] = 0
        subset_counts[choice] += 1
    return subset_counts


def read_data(dataset):
    slate_sizes = []
    slates = []
    choice_sizes = []
    choices = []
    with open(dataset) as f:
        for line in f.readlines():
            slate_str, choice_str = line[:-1].split(";")
            slate = [int(xr) for xr in slate_str.split(" ")]
            choice = [int(xr) for xr in choice_str.split(" ")]
            slate.sort()
            choice.sort()
            slates.append(tuple(slate))
            choices.append(tuple(choice))
            slate_sizes.append(len(slate))
            choice_sizes.append(len(choice))
    return VariableChoiceDataset(slate_sizes, slates, choice_sizes, choices)


def in_H(model, choice):
    return tuple(choice) in list(model.H.keys())


def H_val(model, choice):
    if tuple(choice) in model.H.keys():
        return model.H[tuple(choice)]
    return 0.


def set_H_value(model, choice, val):
    model.H[tuple(choice)] = val


def add_to_H(model, choice_to_add):
    if (in_H(model, choice_to_add)):
        print("Choice Already in H Set")
    set_H_value(model, choice_to_add, 0.)


'''
This function sums Exponentials of all subset of utilities of a given size
'''


def sumexp_util(model, slate, subset_size):
    subsets = list(itertools.combinations_with_replacement(slate, subset_size))
    subsets_utils_sums = list(map(lambda x: np.sum([model.utilities[xr - 1] for xr in x]), subsets))
    subset_H_vals = list(map(lambda x: H_val(model, x), subsets))
    subset_exps = np.exp(np.array([subsets_utils_sums[i] + subset_H_vals[i] for i in range(len(subsets))]))

    return np.sum(subset_exps)


'''
This function calculates gradient for a given model configuration for that data
'''


def gradient_update(model, grad, slate, H_inds, subset_size):
    sum = sumexp_util(model, slate, subset_size)
    if (np.isnan(sum)):
        print("Error! Gradient sum is Nan")
        exit(1)
    subsets = list(itertools.combinations_with_replacement(slate, subset_size))
    subsets_utils_sums = list(map(lambda x: np.sum([model.utilities[xr - 1] for xr in x]), subsets))
    subset_H_vals = list(map(lambda x: H_val(model, x), subsets))
    for i in range(len(subsets)):
        subset, subset_exp = subsets[i], np.exp(subsets_utils_sums[i] + subset_H_vals[i])
        for elem in subset:
            grad[elem - 1] += subset_exp / sum
        if in_H(model, subset):
            grad[H_inds[subset] - 1] += subset_exp / sum
    return None


'''
log likelihood for a given model and data, sum of individual slate/choice dataset
'''


def log_likelihood(model, data):
    ns = len(data.slate_sizes)
    ll = {i: 0 for i in range(1, ns + 1)}
    for i in range(1, ns + 1):
        slate = data.slates[i-1]
        choice = data.choices[i-1]
        size = len(choice)
        max_ind = min(len(model.z), len(slate) - 1)
        ll[i] += np.log(model.z[size - 1] / np.sum(model.z[:max_ind]))
        for item in choice:
            ll[i] += model.utilities[item - 1]
        ll[i] += H_val(model, choice)
        ll[i] -= np.log(sumexp_util(model, slate, size))
    return np.sum(list(ll.values()))


'''
This function learns utilities for a given model
'''


def learn_utilities(model, data):
    n_items = len(model.utilities)
    H_tups = []
    H_inds = {}
    H_vals = []
    for (tup, val) in model.H.items():
        H_tups.append(tup)
        H_vals.append(val)
        H_inds[tup] = n_items + len(H_tups)

    def update_model(x):
        x[np.isnan(np.array(x))] = 0.
        model.utilities = x[:n_items]
        for (tup, val) in zip(H_tups, x[n_items:]):
            set_H_value(model, tup, val)

    def neg_log_likelihood(x):
        update_model(x)
        return -log_likelihood(model, data)

    def gradient(x):
        grad = np.zeros(len(x))
        update_model(x)
        for i in range(len(data.slate_sizes)):
            slate = data.slates[i]
            choice = data.choices[i]
            size = len(choice)
            for item in choice:
                grad[item - 1] -= 1
            if in_H(model, choice):
                grad[H_inds[tuple(choice)] - 1] -= 1
            gradient_update(model, grad, slate, H_inds, size)

        gnorm = np.linalg.norm(grad, ord=None)
        if (gnorm > 10):
            for i in range(len(grad)):
                grad[i] /= gnorm
        return grad

    x0 = np.append(model.utilities, np.array(H_vals))
    res = Optimizer(neg_log_likelihood, x0, method='L-BFGS-B', jac=gradient,
                    options={'disp': True, 'ftol': 1e-3, 'maxfun': 25})
    update_model(res.x)


def learn_size_probs(model, data):
    def neg_log_likelihood(x):
        nll = 0.
        for (slate_size, choice_size) in zip(data.slate_sizes, data.choice_sizes):
            nll -= x[choice_size - 1]
            total = 0.
            max_choice_size = min(slate_size - 1, len(x))
            for i in range(max_choice_size):
                total += np.exp(x[i])
            nll += np.log(total)
        return nll

    def gradient(x):
        grad = np.zeros(len(x))
        for (slate_size, choice_size) in zip(data.slate_sizes, data.choice_sizes):
            if choice_size > 1:
                grad[choice_size - 1] -= 1
            total = 1.
            max_choice_size = min(slate_size - 1, len(x))
            for i in range(1, max_choice_size):
                total += np.exp(x[i])
            grad[1:max_choice_size] += np.exp(np.array(x[1:max_choice_size])) / total
        # gnorm = np.linalg.norm(grad, ord=None)
        # if (gnorm > 10):
        #     for i in range(len(grad)):
        #         grad[i] /= gnorm

        return grad

    x0 = np.zeros(np.max(data.choice_sizes))
    res = Optimizer(neg_log_likelihood, x0, method='L-BFGS-B', jac=gradient,
                    options={'disp': True, 'ftol': 1e-6})
    # res = minOptimizer(func = neg_log_likelihood, x0 = x0,  factr=1e-6, fprime=gradient, iprint=101)
    model.z = np.exp(res.x) / sum(np.exp(res.x))


def initialize_model(data):
    max_choice_size = np.max(data.choice_sizes)
    z = np.ones(max_choice_size)
    utilities = np.zeros(np.max([max(xr) for xr in data.slates]))
    H = {}
    return VariableChoiceModel(z, utilities, H)

def initialize_model_with_H(data, choices_to_add):
    max_choice_size = np.max(data.choice_sizes)
    z = np.ones(max_choice_size)
    utilities = np.zeros(np.max([max(xr) for xr in data.slates]))
    H = {}
    model = VariableChoiceModel(z, utilities, H)
    for choice_to_add in choices_to_add:
        add_to_H(model, choice_to_add)
    return model


def learn_model(model, data):
    learn_size_probs(model, data)
    learn_utilities(model, data)

import numpy as np
from numpy import linalg
import itertools
from scipy.optimize import minimize as Optimizer
from scipy.optimize import fmin_l_bfgs_b as minOptimizer

'''
This Class represents a variable choice dataset in terms of slates, and made choices
'''


class VariableMNLChoiceDataset:
    def __init__(self, slates, choices, slate_choice_pairs, slate_choice_counts, choice_count_index, choice_counts, items, item_counts):
        self.slates = np.array(slates)
        self.choices = np.array(choices)
        self.slate_choice_pair = slate_choice_pairs
        self.slate_choice_count = slate_choice_counts
        self.choice_count_index = choice_count_index
        self.choice_counts = choice_counts
        self.items = items
        self.item_counts  =item_counts


'''
This class represents a variable choice model in terms of Utilities, Z probabilites and correction
factors(H)
'''


class VariableMNLChoiceModel:
    def __init__(self, utilities, item_index):
        self.utilities = utilities
        self.item_index = item_index

    def get_utility(self, item):
        if (item in self.item_index):
            return self.utilities[item]
        else:
            return -np.inf


'''
This Function reads data in given format
'''


def read_data(dataset):
    slates = []
    choices = []
    slate_choice_pairs = []
    slate_choice_counts = []
    choice_count_index = []
    choice_counts = []
    items =[]
    item_counts = []
    with open(dataset) as f:
        for line in f.readlines():
            slate_str, choice_str = line[:-1].split(";")
            slate = [int(xr) for xr in slate_str.split(" ")]
            choice = [int(xr) for xr in choice_str.split(" ")]
            slate.sort()
            choice.sort()
            slates.append(tuple(slate))
            choices.append(tuple(choice))
            for item in choice:
                if item not in items:
                    items.append(item)
                    item_counts.append(0)
                item_counts[items.index(item)]+=1
            if tuple(choice) not in choice_count_index:
                choice_count_index.append(tuple(choice))
                choice_counts.append(0)
            choice_counts[choice_count_index.index(tuple(choice))] += 1
            slate_choice_tuple = tuple([tuple(slate), tuple(choice)])
            if slate_choice_tuple not in slate_choice_pairs:
                slate_choice_pairs.append(slate_choice_tuple)
                slate_choice_counts.append(0)
            slate_choice_counts[slate_choice_pairs.index(slate_choice_tuple)] += 1
    return VariableMNLChoiceDataset(slates, choices, slate_choice_pairs, slate_choice_counts, choice_count_index,
                                    choice_counts, items, item_counts)


'''
This function sums Exponentials of all subset of utilities of a size 1 and 2
'''


def sumexp_util(model, slate):
    slate = list(slate)
    item_exps = [np.exp(model.get_utility(item)) for item in slate]
    return np.nansum(item_exps)


'''
log likelihood for a given model and data, sum of individual slate/choice dataset
'''


def log_likelihood(model, data):
    ns = len(data.slates)
    ll = {i: 0 for i in range(1, ns + 1)}
    for i in range(1, ns + 1):
        slate = data.slates[i - 1]
        choice = data.choices[i - 1]
        slate_all_subset_expsum = sumexp_util(model, slate)
        ll[i] = - len(choice)* np.log(slate_all_subset_expsum)
        for item in choice:
            if item in model.item_index:
                ll[i] += model.get_utility(item)
            else:
                ll[i] += 0

    print('LL Cal', np.nansum(list(ll.values())))
    return np.nansum(list(ll.values()))


'''
This function learns utilities for a given model
'''


def learn_utilities(model, data):
    def update_model(x):
        x[np.isnan(np.array(x))] = 0.
        for i in range(len(x)):
            model.utilities[model.item_index[i]] = x[i]

    def neg_log_likelihood(x):
        update_model(x)
        return -log_likelihood(model, data)

    def gradient(x):
        grad = np.zeros(len(x))
        update_model(x)
        for i in range(len(grad)):
            if model.item_index[i] in data.items:
                grad[i] -= data.item_counts[data.items.index(model.item_index[i])]

        for i, (slate, choice) in enumerate(zip(data.slates, data.choices)):
            slate_all_subset_expsum = sumexp_util(model, slate)
            slate = list(slate)
            slate.sort()
            for item in slate:
                if item in model.item_index:
                    grad[model.item_index.index(item)] += (len(choice)* np.exp(model.get_utility(item))) / slate_all_subset_expsum
        gnorm = np.linalg.norm(grad, ord=None)
        if gnorm > 10:
            for i in range(len(grad)):
                grad[i] /= gnorm
        print('grad Cal', np.sum(grad))
        return grad

    x0 = np.zeros(len(model.item_index))
    res = Optimizer(neg_log_likelihood, x0, method='L-BFGS-B', jac=gradient,
                    options={'disp': True, 'ftol': 1e-3, 'maxfun': 25})
    update_model(res.x)


def initialize_model(data):
    max_item_id = np.max([max(xr) for xr in data.slates])
    utilities = dict.fromkeys(list(range(1, max_item_id+1)), 0)
    return VariableMNLChoiceModel(utilities, list(range(1,max_item_id+1)))
    # utilities = dict.fromkeys(data.items,0)
    # return VariableMNLChoiceModel(utilities, data.items)


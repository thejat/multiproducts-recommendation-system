import numpy as np
from numpy import linalg
import itertools
from scipy.optimize import minimize as Optimizer
from scipy.optimize import fmin_l_bfgs_b as minOptimizer

'''
This Class represents a variable choice dataset in terms of slates, and made choices
'''


class VariableRCMChoiceDataset:
    def __init__(self, slates, choices, slate_choice_pairs, slate_choice_counts, choice_count_index, choice_counts):
        self.slates = np.array(slates)
        self.choices = np.array(choices)
        self.slate_choice_pair = slate_choice_pairs
        self.slate_choice_count = slate_choice_counts
        self.choice_count_index = choice_count_index
        self.choice_counts = choice_counts


'''
This class represents a variable choice model in terms of Utilities, Z probabilites and correction
factors(H)
'''


class VariableRCMChoiceModel:
    def __init__(self, utilities, choice_index):
        self.utilities = utilities
        self.choice_index = choice_index

    def get_utility(self, choice):
        if (choice in self.choice_index):
            return self.utilities[choice]
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
    with open(dataset) as f:
        for line in f.readlines():
            slate_str, choice_str = line[:-1].split(";")
            slate = [int(xr) for xr in slate_str.split(" ")]
            choice = [int(xr) for xr in choice_str.split(" ")]
            slate.sort()
            choice.sort()
            slates.append(tuple(slate))
            choices.append(tuple(choice))
            if tuple(choice) not in choice_count_index:
                choice_count_index.append(tuple(choice))
                choice_counts.append(0)
            choice_counts[choice_count_index.index(tuple(choice))] += 1
            slate_choice_tuple = tuple([tuple(slate), tuple(choice)])
            if slate_choice_tuple not in slate_choice_pairs:
                slate_choice_pairs.append(slate_choice_tuple)
                slate_choice_counts.append(0)
            slate_choice_counts[slate_choice_pairs.index(slate_choice_tuple)] += 1
    return VariableRCMChoiceDataset(slates, choices, slate_choice_pairs, slate_choice_counts, choice_count_index,
                                    choice_counts)


'''
This function sums Exponentials of all subset of utilities of a size 1 and 2
'''


def sumexp_util(model, slate):
    slate = list(slate)
    slate.sort()
    subsets_size_1 = list(itertools.combinations(slate, 1))
    subsets_size_2 = list(itertools.combinations(slate, 2))
    subsets = subsets_size_1 + subsets_size_2
    subset_exps = [np.exp(model.get_utility(tuple(subsets[i]))) for i in range(len(subsets))]
    return np.nansum(subset_exps)


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
        # slate_choice_tuple = tuple([tuple(slate), tuple(choice)])
        if choice in model.choice_index:
            ll[i] = model.get_utility(tuple(choice)) - np.log(slate_all_subset_expsum)
        else:
            ll[i] = 0
        # ll[i] = data.slate_choice_count[data.slate_choice_pair.index(slate_choice_tuple)] * (
        #         np.exp(model.get_utility(tuple(choice))) / slate_all_subset_expsum)
    print('LL Cal', np.nansum(list(ll.values())))
    return np.nansum(list(ll.values()))


'''
This function learns utilities for a given model
'''


# def gradient_test(model, data):
#     grad = np.zeros(len(model.choice_index))
#     # update_model(x)
#     for i in range(len(grad)):
#         if model.choice_index[i] in data.choice_count_index:
#             grad[i] += data.choice_counts[data.choice_count_index.index(model.choice_index[i])]
#
#     for i, (slate, choice) in enumerate(zip(data.slates, data.choices)):
#         slate_all_subset_expsum = sumexp_util(model, slate)
#         slate = list(slate)
#         slate.sort()
#         subsets_size_1 = list(itertools.combinations(slate, 1))
#         subsets_size_2 = list(itertools.combinations(slate, 2))
#         subsets = subsets_size_1 + subsets_size_2
#         for subset in subsets:
#             N_sc = data.slate_choice_count[data.slate_choice_pair.index(tuple([tuple(slate), choice]))]
#             grad[model.choice_index.index(tuple(subset))] -= (N_sc * np.exp(
#                 model.utilities[subset])) / slate_all_subset_expsum
#     # gnorm = np.linalg.norm(grad, ord=None)
#     # if (gnorm > 10):
#     #     for i in range(len(grad)):
#     #         grad[i] /= gnorm
#
#     return grad


def learn_utilities(model, data):
    def update_model(x):
        x[np.isnan(np.array(x))] = 0.
        for i in range(len(x)):
            model.utilities[model.choice_index[i]] = x[i]

    def neg_log_likelihood(x):
        update_model(x)
        return -log_likelihood(model, data)

    def gradient(x):
        grad = np.zeros(len(x))
        update_model(x)
        for i in range(len(grad)):
            if model.choice_index[i] in data.choice_count_index:
                grad[i] -= data.choice_counts[data.choice_count_index.index(model.choice_index[i])]

        for i, (slate, choice) in enumerate(zip(data.slates, data.choices)):
            slate_all_subset_expsum = sumexp_util(model, slate)
            slate = list(slate)
            slate.sort()
            subsets_size_1 = list(itertools.combinations(slate, 1))
            subsets_size_2 = list(itertools.combinations(slate, 2))
            subsets = subsets_size_1 + subsets_size_2
            for subset in subsets:
                if subset in model.choice_index:
                    # n_sc = data.slate_choice_count[data.slate_choice_pair.index(tuple([tuple(slate), choice]))]
                    grad[model.choice_index.index(tuple(subset))] += (np.exp(
                        model.get_utility(subset))) / slate_all_subset_expsum
        gnorm = np.linalg.norm(grad, ord=None)
        if gnorm > 10:
            for i in range(len(grad)):
                grad[i] /= gnorm
        print('grad Cal', np.sum(grad))
        return grad

    x0 = np.zeros(len(model.choice_index))
    res = Optimizer(neg_log_likelihood, x0, method='L-BFGS-B', jac=gradient,
                    options={'disp': True, 'ftol': 1e-3, 'maxfun': 25})
    # min_nll = neg_log_likelihood(x0)
    # f_tol = 10e-3
    # lr = 100
    # x = x0
    # while 1:
    #     grad_x = np.array(gradient(x))
    #     x -= lr * np.array(grad_x)
    #     nll_val = neg_log_likelihood(x)
    #     if min_nll < nll_val:
    #         lr = lr / 2
    #     if np.absolute(nll_val - min_nll) < f_tol:
    #         break
    #     min_nll = nll_val
    #     print(np.sum(x), np.sum(grad_x), nll_val)
    #
    # return None
    update_model(res.x)

    # x0 = np.zeros(np.max(data.choice_sizes))
    # res = Optimizer(neg_log_likelihood, x0, method='L-BFGS-B', jac=gradient,
    #                 options={'disp': True, 'ftol': 1e-6})
    # res = minOptimizer(func = neg_log_likelihood, x0 = x0,  factr=1e-6, fprime=gradient, iprint=101)
    # model.z = np.exp(res.x) / sum(np.exp(res.x))


def initialize_model(data):
    unique_choices = []
    unique_items = []
    for i in range(len(data.choices)):
        slate, choice = tuple(data.slates[i]), tuple(data.choices[i])
        if choice not in unique_choices:
            unique_choices.append(choice)
        for item in slate:
            if item not in unique_items:
                unique_items.append(item)

    # total_items = len(unique_items)
    utilities = dict.fromkeys(unique_choices, 0)
    return VariableRCMChoiceModel(utilities, unique_choices)

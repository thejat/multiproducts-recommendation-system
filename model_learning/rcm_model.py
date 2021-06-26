import numpy as np
from model_learning.ucm_model import get_subset_counts, UniversalChoiceDataset


class RCMChoiceModel:
    def __init__(self, probs):
        self.probs = probs

'''
This function reads data from corresponding datafile into UniversalChoiceDataset
'''


def read_data(datafilepath, delimiter=' '):
    choices, sizes = [], []
    with open(datafilepath, 'r') as f:
        for line in f.readlines():
            choice = [int(item) for item in line[:-1].split(delimiter)]
            if(len(choice)>1):
                if(choice[0]==choice[1]):
                    choice = choice[:1]
            choice.sort()
            choices.append(tuple(choice))
            sizes.append(len(choice))
    return UniversalChoiceDataset(sizes, choices)


def log_likelihood(model, data):
    subsets_counts = get_subset_counts(data)
    lls = {k: subsets_counts[k] * np.log(model.probs[k]) for k in subsets_counts.keys() if k in model.probs.keys()}
    return np.nansum(list(lls.values()))


def initialize_model(data):
    subsets_counts = get_subset_counts(data)
    total_subsets = len(data.sizes)
    subset_probs = {k: subsets_counts[k] / total_subsets for k in subsets_counts.keys()}

    return RCMChoiceModel(subset_probs)

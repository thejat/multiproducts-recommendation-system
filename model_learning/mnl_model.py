import numpy as np

'''
MNLChoice Model: This is simple model based on frequency of items selected in given dataset
'''


class MNLChoiceModel:
    def __init__(self, probs):
        self.probs = probs


def log_likelihood(model, data):
    item_counts = {k: 0 for k in model.probs.keys()}
    for choice in data.choices:
        for item in choice:
            item_counts[item] += 1
    lls = {k: (item_counts[k] * np.log(model.probs[k])) for k in model.probs.keys()}

    return np.nansum(list(lls.values()))


def initialize_model(data):
    unique_items = []
    for choice in data.choices:
        for item in choice:
            if item not in unique_items:
                unique_items.append(item)
    # total_items = max([max(choice) for choice in data.choices])
    item_counts = dict.fromkeys(unique_items,1)
    # item_counts = {i: 1 for i in range(1, total_items + 1)}
    for choice in data.choices:
        for item in choice:
            item_counts[item] += 1
    total = sum(list(item_counts.values()))
    item_probs = {k: (item_counts[k] / total) for k in item_counts.keys()}
    return MNLChoiceModel(item_probs)



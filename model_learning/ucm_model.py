import numpy as np

'''
Represent a universal choice dataset as a array of sizes (lengths of choices)
and a array of all items in all choices.
'''


class UniversalChoiceDataset:
    def __init__(self, sizes, choices):
        self.sizes = sizes
        self.choices = choices


'''
Probabilistic universal subset choice model.
-z: is a vector of length max_size with the probability of choosing
 a size-k subset being z[k]
-probs: are the item probabilities
-gammas: is a vector of length max_size of normalization parameters
-H: is a vector of length max_size where each element
 is a dictionary that maps a choice to a probability.
'''


class UniversalChoiceModel:
    def __init__(self, z, probs, gammas, H, item_counts, subset_counts, size_counts):
        self.z = z
        self.probs = probs
        self.gammas = gammas
        self.H = H
        # Some data to make updating computations faster
        # how often each item appears not in a hot set
        self.item_counts = item_counts
        # how many times each subset appears
        self.subset_counts = subset_counts
        # number of times choice set of each size appears
        self.size_counts = size_counts


'''
This function gets subsets and their counts for given UniversalChoiceDataset
'''


def get_subset_counts(data):
    subset_counts = {}
    for i in range(len(data.sizes)):
        choice = tuple(data.choices[i])
        if choice not in subset_counts.keys():
            subset_counts[choice] = 0
        subset_counts[choice] += 1
    return subset_counts


'''
This is another helper function to return subsets and counts for a UniversalChoiceDataset 
in seperate variables.
'''


def subsets_and_counts(data):
    subsets = []
    counts = []
    for (subset, count) in get_subset_counts(data).items():
        subsets.append(subset)
        counts.append(count)
    return subsets, counts


'''
This function reads data from corresponding datafile into UniversalChoiceDataset
'''


def read_data(datafilepath, delimiter=' '):
    choices, sizes = [], []
    with open(datafilepath, 'r') as f:
        for line in f.readlines():
            choice = [int(item) for item in line[:-1].split(delimiter)]
            choice.sort()
            choices.append(tuple(choice))
            sizes.append(len(choice))
    return UniversalChoiceDataset(sizes, choices)


'''
These set of function helps to get probability for correction factors(H) from model
'''


def in_H(model, choice): return tuple(choice) in model.H.keys()


def H_prob(model, choice): return model.H[tuple(choice)]


'''
This function calculates log likehood for a Given UCM Model, and given subsets and their counts
'''


def log_likelihood_counts(model, subsets, counts):
    ns = len(subsets)
    all_lls = np.zeros_like(subsets)
    for i in range(ns):
        choice = subsets[i]
        size = len(choice)
        ll = np.log(model.z[size])
        if in_H(model, choice):
            ll += np.log(H_prob(model, choice))
        else:
            ll += np.log(model.gammas[len(choice)])
            for item in choice:
                if model.probs[item] == 0:
                    ll = 0
                    break
                ll += np.log(model.probs[item])
        all_lls[i] = counts[i] * ll
    return np.nansum(all_lls)


'''
This is a wrapper function on log_likelihood to get LL for a dataset
'''


def log_likelihood(model, data):
    subsets, counts = subsets_and_counts(data)
    return log_likelihood_counts(model, subsets, counts)


'''
This function calculates proportionality constants(For each subset size)
'''


def normalization_values(max_size, H, item_probs):
    gammas = {i: 0 for i in range(1, max_size + 1)}

    # For 1 size subset, no proportionality constant
    gammas[1] = 1

    if max_size == 1:
        return gammas

    # Normalization for Size 2 choice probalities
    base_probs = {i: 0 for i in range(1, max_size + 1)}
    H_probs = {i: 0 for i in range(1, max_size + 1)}

    for (subset, val) in H.items():
        ns = len(subset)
        base_probs[ns] += np.prod([item_probs[item] for item in subset])
        H_probs[ns] += val

    '''
    The Logic for this normalization comes from following,
    Let A = SUM(i's) pi**2
    and B = SUM(i,j, i!=j) pi.pj
    Then SUM(i, j) pipj = A+2B, which sum of probability of choosing any 2 sized subset in order.
    i.e A+2B = 1
    Thus B = (1-A)/2
    '''
    sum_pi2 = np.sum([p ** 2 for p in item_probs.values()])
    sum_pipj = (1 - sum_pi2) / 2
    # print(sum_pi2, sum_pipj)
    gammas[2] = (1 - H_probs[2]) / (sum_pipj + sum_pi2 - base_probs[2])

    if max_size == 2:
        return gammas

    # normalization for size-3 choice probabilities
    sum_pi3 = np.sum([p ** 3 for p in item_probs.values()])
    sum_pipjpj = sum_pi2 - sum_pi3
    sum_pipjpk = (1 - sum_pi3 - 3 * sum_pipjpj) / 6
    gammas[3] = (1 - H_probs[3]) / (sum_pipjpk + sum_pipjpj + sum_pi3 - base_probs[3])

    if max_size == 3:
        return gammas
    # normalization for size-4 choice probabilities
    sum_pi4 = np.sum([p ** 4 for p in item_probs.values()])
    sum_pi2pj2 = (sum_pi2 ** 2 - sum_pi4) / 2
    sum_pipj3 = sum_pi3 - sum_pi4
    sum_pipjpk2 = (sum_pi2 - sum_pi4 - 2 * sum_pipj3 - 2 * sum_pi2pj2) / 2
    sum_pipjpkpl = (1 - sum_pi4 - 4 * sum_pipj3 - 6 * sum_pi2pj2 - 12 * sum_pipjpk2) / 24
    gammas[4] = (1 - H_probs[4]) / (sum_pipjpkpl + sum_pi2pj2 + sum_pipj3 + sum_pipjpk2 + sum_pi4 - base_probs[4])

    if max_size == 4:
        return gammas
    # normalization for size-5 choice probabilities
    sum_pi5 = sum([p ** 5 for p in item_probs.values()])
    sum_pipj4 = sum_pi4 - sum_pi5
    sum_pi2pj3 = sum_pi2 * sum_pi3 - sum_pi5
    sum_pipjpk3 = (sum_pi3 - sum_pi5 - 2 * sum_pipj4 - sum_pi2pj3) / 2
    sum_pipj2pk2 = (sum_pi2 ** 2 - sum_pipj4 - 2 * sum_pi2pj3 - sum_pi5) / 2
    sum_pipjpkpl2 = sum_pi2 * sum_pipjpk - sum_pipjpk3
    sum_pipjpkplpm = (1.0 - sum_pi5 - 5 * sum_pipj4 - 20 * sum_pipjpk3 - 60 * sum_pipjpkpl2 -
                      30 * sum_pipj2pk2 - 10 * sum_pi2pj3) / 120
    gammas[5] = (1 - H_probs[5]) / (
            sum_pipjpkplpm + sum_pi5 + sum_pipj4 + sum_pi2pj3 + sum_pipjpk3 + sum_pipjpkpl2 + sum_pipj2pk2 -
            base_probs[5])

    # TODO: support larger sizes.  We really shouldn't hard code the above.
    if max_size > 5:
        print("Support only for choices of size <= 5 items")
        exit(1)

    return gammas


'''
This function restructures probabilities once we add a set to H, based on correction factor model in UCM paper
'''


def add_to_H(model, choice_to_add):
    if in_H(model, tuple(choice_to_add)):
        print("choice ", str(choice_to_add), "already in model..")
        return

    # Update H probability
    choice_tup = tuple(choice_to_add)
    choice_count = model.subset_counts[choice_tup]
    model.H[choice_tup] = choice_count / (model.size_counts[len(choice_to_add)])

    # Update each item count based on H Update
    for item in choice_to_add:
        model.item_counts[item] -= choice_count

    total = np.sum(list(model.item_counts.values()))
    for i in model.item_counts.keys():
        model.probs[i] = model.item_counts[i] / total

    # Update Normalization parameters
    model.gammas = normalization_values(len(model.gammas), model.H, model.probs)

    return None


'''
This is main function which initializes model based on a UniversalDataset assuming no subset choice in H..
'''


def initialize_model(data):
    max_size = max(data.sizes)

    # find fix size probabilities
    z = {i: 0 for i in range(1, max_size + 1)}
    size_counts = {i: 0 for i in range(1, max_size + 1)}
    # Calculate size frequencies
    for size in data.sizes:
        size_counts[size] += 1
    # Calculate z's
    for key in size_counts:
        z[key] = size_counts[key] / len(data.sizes)

    # item probabilities Initialized with empirical fraction of choice sets it appears in
    total_items = max([max(choice) for choice in data.choices])
    item_counts = {i: 0 for i in range(1, total_items + 1)}

    for choice in data.choices:
        for item in choice:
            item_counts[item] += 1
    total = sum(item_counts.values())
    item_probs = {k: (item_counts[k] / total) for k in item_counts.keys()}

    # Initialize H set
    H = {}

    # Initialize Normalization constants(gammas)
    gammas = normalization_values(max_size, H, item_probs)

    # counts of how many times each choice set appears
    subset_counts = get_subset_counts(data)

    return UniversalChoiceModel(z, item_probs, gammas, H, item_counts, subset_counts, size_counts)


def initialize_model_with_H(data, choices_to_add):
    max_size = max(data.sizes)
    # find fix size probabilities
    z = {i: 0 for i in range(1, max_size + 1)}
    size_counts = {i: 0 for i in range(1, max_size + 1)}
    # Calculate size frequencies
    for size in data.sizes:
        size_counts[size] += 1
    # Calculate z's
    for key in size_counts:
        z[key] = size_counts[key] / len(data.sizes)

    unique_items = []
    for choice in data.choices:
        for item in choice:
            if item not in unique_items:
                unique_items.append(item)
    # total_items = max([max(choice) for choice in data.choices])
    item_counts = dict.fromkeys(unique_items, 1)

    for choice in data.choices:
        if(choice not in choices_to_add):
            for item in choice:
                item_counts[item] += 1

    total = sum(item_counts.values())
    item_probs = {k: (item_counts[k] / total) for k in item_counts.keys()}

    # counts of how many times each choice set appears
    subset_counts = get_subset_counts(data)

    # Initialize H set
    H={}
    for choice_to_add in choices_to_add:
        H[choice_to_add] = subset_counts[choice_to_add] / (size_counts[len(choice_to_add)])

    # Initialize Normalization constants(gammas)
    gammas = normalization_values(max_size, H, item_probs)

    return UniversalChoiceModel(z, item_probs, gammas, H, item_counts, subset_counts, size_counts)

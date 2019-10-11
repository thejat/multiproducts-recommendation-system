import numpy as np
import time
import random
import collections
from itertools import combinations
from synthetic_models.utils import generate_instance, set_char_from_ast


# generate Universal Choice Set model instance


def generate_universal_choice_model(price_range, prod, n_Hset_count, maxHset_size=2, max_purchase_size=2,
                                    include_no_purchase=True):
    # Get some H sets
    start_time = time.time()
    # if n_Hset_count is None:
    #
    try:
        assert (maxHset_size <= max_purchase_size)
    except AssertionError:
        print(f"Error:Max Hset Size:{maxHset_size} is greater than max purchase size:{max_purchase_size}...")
        exit(0)

    p, v = generate_instance(price_range, prod, genMethod=None, iterNum=None)
    if not include_no_purchase:
        v[0] = 0
    Hsets, W = get_H_sets(prod, n_Hset_count, maxHset_size, start_time, include_no_purchase)
    gamma, prob_unnormalized = get_gamma(max_purchase_size, prod, v, Hsets, W)
    probs, probs_select_size = get_probabilities_given_gamma(max_purchase_size, prod, gamma, prob_unnormalized, v)

    return {'probs': probs, 'probs_select_size': probs_select_size, 'gamma': gamma, 'Hsets': Hsets, 'W': W, 'p': p,
            'v': v, 'max_Hset_size': maxHset_size, 'max_purchase_size': max_purchase_size}


def get_H_sets(prod, n_Hset_count, max_Hset_size, start_time, include_no_purchase):
    nHsets = 0
    Hsets = []
    W = collections.OrderedDict({})
    nHsets_temp = 0

    # if max_Hset_size > 1:
    #     while nHsets < n_Hset_count:
    #         nHsets_temp += 1
    #         if nHsets_temp % (n_Hset_count * 50) == 0:
    #             print('Time elapsed: {0}. nHsets:{1}'.format(time.time() - start_time, nHsets))
    #         if include_no_purchase:
    #             temp = random.randint(0, 2 ** prod - 1)
    #         else:
    #             temp = random.randint(1, 2 ** prod - 1)
    #         set_char_vector = tuple([int(x) for x in format(temp, '0' + str(prod) + 'b')])
    #         if (sum(set_char_vector) <= max_Hset_size) & (sum(set_char_vector) > 1):
    #             Hsets.append(set_char_vector)
    #             nHsets += 1
    #             # W[set_char_vector] = 100 * random.random()
    #             W[set_char_vector] = random.random()
    all_subsets = np.array([(0, 0)] + list(combinations(list(range(1, prod)), 2)))
    np.random.shuffle(all_subsets)

    Hsets_idx = np.random.choice(len(all_subsets), n_Hset_count)

    for idx in Hsets_idx:
        set_char_vector = set_char_from_ast(all_subsets[idx], prod)
        Hsets.append(set_char_vector)
        W[set_char_vector] = random.random()
    # # TODO: Random Code for Verificatio, Remove it
    # Hsets.append(tuple([0] * 5))
    # nHsets += 1
    # # W[tuple([0]*5)] = 100 * random.random()
    # W[tuple([0] * 5)] = random.random()
    # print(collections.Counter([sum(x) for x in Hsets]))
    return Hsets, W


def get_gamma(max_purchase_size, prod, v, Hsets, W):
    # Obtain Gamma values
    gamma = collections.OrderedDict({})
    prob_unnormalized = {}
    for size in range(1, max_purchase_size + 1):
        gamma[size] = 0
        prob_unnormalized[size] = {}
        # print(list(combinations(range(prod + 1), size)))
        subsets = [tuple([0] * size)] + list(combinations(range(1, prod + 1), size))
        # print(subsets)
        for ast in subsets:
            set_char_vector = set_char_from_ast(ast, prod)
            prob_temp = 1
            for x in ast:
                prob_temp *= v[x]
            if (set_char_vector in Hsets) & (size > 1):
                q_temp = prob_temp * np.exp(W[set_char_vector]) - prob_temp
            else:
                q_temp = 0
            gamma[size] += prob_temp + q_temp
            prob_unnormalized[size][set_char_vector] = prob_temp + q_temp
        gamma[size] = 1.0 / gamma[size]

    # print('gamma',gamma)
    return gamma, prob_unnormalized


def get_probabilities_given_gamma(max_purchase_size, prod, gamma, prob_unnormalized, v):
    # Obtain probabilities
    probs = collections.OrderedDict({x: {} for x in range(1, max_purchase_size + 1)})
    for size in range(1, max_purchase_size + 1):
        tempsum = 0
        subsets = [tuple([0] * size)] + list(combinations(range(1, prod + 1), size))
        for ast in subsets:
            set_char_vector = set_char_from_ast(ast, prod)
            probs[size][set_char_vector] = gamma[size] * prob_unnormalized[size][set_char_vector]

            tempsum += probs[size][set_char_vector]
        # print('size',size,'sum of probabilities conditional on size',tempsum)
        probs[size][set_char_vector] += 1 - tempsum  # minor correction hack so probs sum to 1

    # get size selection probabilities
    probs_select_size = collections.OrderedDict({0: v[0]})

    temp_total = 0
    for size in range(1, max_purchase_size + 1):
        probs_select_size[size] = .1 / size  # random.random()
        temp_total += probs_select_size[size]
    for size in range(1, max_purchase_size + 1):
        # probs_select_size[size] = (1.0 - v[0]) * probs_select_size[size] / temp_total
        # TODO: Setting Z0 to be zero, Temporary Change.
        probs_select_size[size] = probs_select_size[size] / temp_total
    probs_select_size[0] += 1 - sum(probs_select_size.values())  # minor correction hack so probs sum to 1

    return probs, probs_select_size

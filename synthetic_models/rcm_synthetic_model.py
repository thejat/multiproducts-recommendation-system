import numpy as np
from synthetic_models.utils import generate_instance


def generate_two_restricted_choice_model(price_range, prod, scaling_v0=1, scaling_v2=1, prob_v0=None):
    p, v = generate_instance(price_range, prod, genMethod=None, iterNum=None)

    v = scaling_v0 * v

    v2 = {}
    for i in range(1, prod + 1):
        for j in range(i + 1, prod + 1):
            v2[(i, j)] = np.random.rand() * scaling_v2
            v2[(j, i)] = v2[(i, j)]
    # Set V[0] such that prob v0 is prob_v0
    if prob_v0 is not None:
        v_sum = sum(v) + sum(v2.values()) - v[0]
        v[0] = prob_v0 * (v_sum) / (1 - prob_v0)

    return {'p': p, 'v': v, 'v2': v2}


def generate_derived_rcm_choice_model(rcm_model, num_products=None, prob_v0=0.1):
    product_ids = np.array(list(rcm_model['p'].keys()))
    selected_products_ids = product_ids
    if num_products is not None:
        selected_products_ids = np.random.choice(product_ids, num_products, replace=False)

    new_rcm_model = {'product_ids': selected_products_ids}
    prices = [0] * (len(selected_products_ids) + 1)
    v = [0] * (len(selected_products_ids) + 1)
    for i, product_id in enumerate(selected_products_ids):
        prices[i + 1] = rcm_model['p'][product_id]
        v[i + 1] = rcm_model['v'][product_id]
    v[0] = np.random.beta(1, 5) * max(v)
    v2 = {}
    for key, value in rcm_model['v2'].items():
        if (key[0] in selected_products_ids) & (key[1] in selected_products_ids):
            new_key = tuple([np.where(selected_products_ids == item)[0][0] + 1 for item in key])
            v2[new_key] = value
            v2[tuple([new_key[1], new_key[0]])] = value

    v2_filled_keys = v2.keys()
    for i in range(1, len(v)):
        for j in range(i+1, len(v)):
            if tuple([i, j]) not in v2_filled_keys:
                v2[tuple([i, j])], v2[tuple([j, i])] = 0, 0

    # Set V[0] such that prob v0 is prob_v0
    if prob_v0 is not None:
        v_sum = sum(v) + sum(v2.values()) - v[0]
        v[0] = prob_v0 * (v_sum) / (1 - prob_v0)

    new_rcm_model.update({'p': prices, 'v': v, 'v2': v2})
    return new_rcm_model

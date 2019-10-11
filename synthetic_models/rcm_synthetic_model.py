import numpy as np
from synthetic_models.utils import generate_instance


def generate_two_restricted_choice_model(price_range, prod, scaling_v0=1, scaling_v2=1):
    p, v = generate_instance(price_range, prod, genMethod=None, iterNum=None)

    v = scaling_v0 * v

    v2 = {}
    for i in range(1, prod + 1):
        for j in range(i + 1, prod + 1):
            v2[(i, j)] = np.random.rand() * scaling_v2
            v2[(j, i)] = v2[(i, j)]

    return {'p': p, 'v': v, 'v2': v2}



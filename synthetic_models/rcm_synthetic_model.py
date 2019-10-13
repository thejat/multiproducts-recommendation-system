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
    #Set V[0] such that prob v0 is prob_v0
    if prob_v0 is not None:
        v_sum = sum(v)+sum(v2.values()) - v[0]
        v[0] = prob_v0*(v_sum)/(1-prob_v0)

    return {'p': p, 'v': v, 'v2': v2}



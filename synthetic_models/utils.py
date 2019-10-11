import numpy as np
import random


def set_char_from_ast(ast, prod):
    set_char_vector = [0] * prod
    # if type(ast) is int:
    if np.sum(ast) == 0:
        return tuple(set_char_vector)

    if isinstance(ast, (int, np.integer)):
        set_char_vector[ast - 1] = 1
    else:
        for x in ast:
            set_char_vector[x - 1] = 1
    return tuple(set_char_vector)


def ast_from_set_char(set_char_vector):
    ast = []
    for idx in range(len(set_char_vector)):
        if abs(set_char_vector[idx] - 1) < 1e-4:
            ast.append(idx + 1)
    return tuple(ast)


def generate_instance(price_range, prod, genMethod=None, iterNum=None):
    # p = np.ones(prod) #
    p = price_range * np.random.beta(1, 1, prod)
    p = np.insert(p, 0, 0)  # inserting 0 as the first element to denote the price of the no purchase option

    # generating the customer preference vector, we don't care that it is in 0,1.
    # Want it away from 0 for numeric. stability.

    # v is a prod+1 length vector as the first element signifies the customer preference for the no purchase option
    v = np.random.beta(1, 5, prod + 1) + 1e-3

    # Ensure that there are no duplicate entires in v - required for Static-MNL.
    u, indices = np.unique(v, return_inverse=True)

    while not (len(u) == prod + 1) or abs(v[0]) < 1e-3:
        if abs(v[0]) < 1e-3:
            v[0] = np.random.rand(1) + 1e-3
            u, indices = np.unique(v, return_inverse=True)
        extraSize = prod + 1 - len(u)
        newEnt = np.random.rand(extraSize) + 1e-3
        v = np.concatenate((u, newEnt))
        u, indices = np.unique(v, return_inverse=True)

    # print("instance max price:",max(p))
    return p, v


def generate_instance_general(price_range, prod, C=None, genMethod=None, iterNum=None, lenFeas=None, real_data=None):
    # arbitrary sets
    if lenFeas is None:
        nsets = int(prod ** 2)
    else:
        nsets = lenFeas

    # synthetic
    feasibles = []
    if C is None:
        C = 0
        flag_c_discover = True
    else:
        flag_c_discover = False
    for i in range(nsets):
        temp = random.randint(1, 2 ** prod - 1)
        temp2 = [int(x) for x in format(temp, '0' + str(prod) + 'b')]
        set_char_vector = np.asarray(temp2)
        if flag_c_discover:
            C = max(C, np.sum(set_char_vector))
            feasibles.append(set_char_vector)
        else:
            if sum(set_char_vector) <= C:
                feasibles.append(set_char_vector)
            # else:
        # 	print('set_char_vec len',sum(set_char_vector))

    p, v = generate_instance(price_range, prod, genMethod, iterNum)

    return p, v, feasibles, int(C), prod

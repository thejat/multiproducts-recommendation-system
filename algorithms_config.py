"""
Following is list of algorithms, and their variations used for optimization.
"""


BINSEARCHIMPROVED_QIP_MTHREAD = {
    'solution_id': 'binSearchImproved_qip_approx_multithread',
    'algo': 'binary-search-improved',
    'comparison_function': 'qip-approx-mthread',
    'QIPApprox_input': 'tmp/ApproxQIP_input.txt',
    'QIPApprox_output': 'tmp/ApproxQIP_output.txt',
    'heuristic_list': ['BURER2002', 'DUARTE2005', 'LAGUNA2009HCE', 'FESTA2002VNS'],
    'time_multiplier': 0.0001,
    'MQLib_dir': '../MQLib/'
}

"""
Binary Search with MIPS Formulation using KNN for Comparision Step
"""
BINSEARCH_NN_EXACT = {
    'solution_id': 'binSearch_nn-exact',
    'algo': 'binary-search',
    'comparison_function': 'nn-exact'
}

"""
Binary Search with MIPS Formulation using LSH for Comparision Step
"""
BINSEARCH_NN_APPROX = {
    'solution_id': 'binSearch_nn-approx',
    'algo': 'binary-search',
    'comparison_function': 'nn-approx'
}

"""
Binary Search with Solving a Quadratic Integer Programming using cplex module
"""
BINSEARCH_QIP_EXACT = {
    'solution_id': 'binSearch_qip_exact',
    'algo': 'binary-search',
    'comparison_function': 'qip-exact'
}

"""
Binary Seach with Solving QUBO Problem with MQLib Heuristics, using multiple heuristics in parallel at comparision step
"""
BINSEARCH_QIP_MTHREAD = {
    'solution_id': 'binSearch_qip_approx_multithread',
    'algo': 'binary-search',
    'comparison_function': 'qip-approx-mthread',
    'QIPApprox_input': 'tmp/ApproxQIP_input.txt',
    'QIPApprox_output': 'tmp/ApproxQIP_output.txt',
    'heuristic_list': ['BURER2002', 'DUARTE2005', 'LAGUNA2009HCE', 'FESTA2002VNS'],
    'time_multiplier': 0.0001,
    'MQLib_dir':'../MQLib/'
}

"""
Solving Exact Formulation for Mixed Integer Programming
"""
MIXED_IP = {
    'solution_id': 'mixed_ip',
    'algo': 'mixed-ip'
}

"""
ADXOPT with using 1 product for exchange/remove/add
"""
ADXOPT1_PRODUCTS = {
    'solution_id': 'adxopt_products',
    'algo': 'adxopt1'
}

"""
ADXOPT with using 2 product sets for exchange/remove/add
"""
ADXOPT2_SETS = {
    'solution_id': 'adxopt_sets',
    'algo': 'adxopt2'
}
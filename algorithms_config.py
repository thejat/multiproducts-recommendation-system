"""
Following is list of algorithms, and their variations used for optimization.
"""

"""
Binary Seach with Solving QUBO Problem with MQLib Heuristics, using multiple heuristics in parallel at comparision step,
and subpartitioning problem using spectral clustering
"""
BINSEARCHIMPROVED_QIP_MTHREAD_SPC = {
    'solution_id': 'binSearch_qip_approx_multithread_spc',
    'algo': 'binary-search-improved',
    'clusters_allowed': True,
    'comparison_function': 'qip-approx-spc',
    'QIPApprox_input': 'tmp/ApproxQIP_input.txt',
    'QIPApprox_output': 'tmp/ApproxQIP_output.txt',
    'heuristic_list': ['BURER2002', 'DUARTE2005', 'LAGUNA2009HCE', 'FESTA2002VNS'],
    'time_multiplier': 0.0001,
    'max_repeat_counter': 3,
    'MQLib_dir': '../MQLib/'
}

"""
Binary Seach with Solving QUBO Problem with MQLib Heuristics, using multiple heuristics in parallel at comparision step,
and subpartitioning problem using spectral clustering
"""
BINSEARCH_QIP_MTHREAD_SPC = {
    'solution_id': 'binSearch_qip_approx_multithread_spc',
    'algo': 'binary-search',
    'clusters_allowed': True,
    'comparison_function': 'qip-approx-spc',
    'QIPApprox_input': 'tmp/ApproxQIP_input.txt',
    'QIPApprox_output': 'tmp/ApproxQIP_output.txt',
    'heuristic_list': ['BURER2002', 'DUARTE2005', 'LAGUNA2009HCE', 'FESTA2002VNS'],
    'time_multiplier': 0.0001,
    'max_repeat_counter': 3,
    'MQLib_dir': '../MQLib/'
}

"""
Binary Seach with Solving QUBO Problem with MQLib Heuristics, using multiple heuristics in parallel at comparision step,
 along with constraint on num of products in assortment
"""

BINSEARCH_QIP_MTHREAD_CONSTRAINED = {
    'solution_id': 'binSearch_qip_approx_multithread_constrained',
    'algo': 'binary-search',
    'constraints_allowed': True,
    'comparison_function': 'qip-approx',
    'QIPApprox_input': 'tmp/ApproxQIP_input.txt',
    'QIPApprox_output': 'tmp/ApproxQIP_output.txt',
    'heuristic_list': ['BURER2002', 'DUARTE2005', 'LAGUNA2009HCE', 'FESTA2002VNS'],
    'time_multiplier': 0.0001,
    'max_repeat_counter': 3,
    'MQLib_dir': '../MQLib/'
}

BINSEARCHIMPROVED_QIP_MTHREAD = {
    'solution_id': 'binSearchImproved_qip_approx_multithread',
    'algo': 'binary-search-improved',
    'comparison_function': 'qip-approx',
    'QIPApprox_input': 'tmp/ApproxQIP_input.txt',
    'QIPApprox_output': 'tmp/ApproxQIP_output.txt',
    'heuristic_list': ['BURER2002', 'DUARTE2005', 'LAGUNA2009HCE', 'FESTA2002VNS'],
    'time_multiplier': 0.0001,
    'max_repeat_counter': 4,
    'MQLib_dir': '../MQLib/'
}

"""
Binary Seach with Solving QUBO Problem with MQLib Heuristics, using multiple heuristics in parallel at comparision step
"""
BINSEARCH_QIP_MTHREAD = {
    'solution_id': 'binSearch_qip_approx_multithread',
    'algo': 'binary-search',
    'comparison_function': 'qip-approx',
    'QIPApprox_input': 'tmp/ApproxQIP_input.txt',
    'QIPApprox_output': 'tmp/ApproxQIP_output.txt',
    'heuristic_list': ['BURER2002', 'DUARTE2005', 'LAGUNA2009HCE', 'FESTA2002VNS'],
    'time_multiplier': 0.0001,
    'max_repeat_counter': 3,
    'MQLib_dir': '../MQLib/'
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

"""
Revenue Ordered Assortment
"""
REVENUE_ORDERED = {
    'solution_id': 'revenue_ordered',
    'algo': 'revenue-ordered'
}

"""
Revenue Ordered Assortment MNL Style
"""
MNL_REVENUE_ORDERED = {
    'solution_id': 'mnl_revenue_ordered',
    'algo': 'mnl-revenue-ordered'
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

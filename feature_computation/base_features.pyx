# cython: language_level=3

cimport cython
from feature_computation import balance_features, graph_features, array_stats
from libc.math cimport abs


'''

Main file to control extraction of features

'''

@cython.boundscheck(False)
@cython.wraparound(False)
def write_stats(list l, str name, dict features_dict):
    cdef double l_mean, l_coeff, l_min, l_max
    l_mean, l_coeff, l_min, l_max = array_stats.get_stats(l)

    features_dict[name + "_mean"] = l_mean
    features_dict[name + "_coeff"] = l_coeff
    features_dict[name + "_min"] = l_min
    features_dict[name + "_max"] = l_max

@cython.boundscheck(False)
@cython.wraparound(False)
def write_entropy_discrete(list l, int number_outcomes, str name, dict features_dict):
    cdef double entropy
    entropy = array_stats.scipy_entropy_discrete(l, number_outcomes)
    features_dict[name + "_entropy"] = entropy

@cython.boundscheck(False)
@cython.wraparound(False)
def write_entropy_continuous(list l, str name, dict features_dict):
    cdef double entropy
    entropy = array_stats.scipy_entropy_continous(l)
    features_dict[name + "_entropy"] = entropy

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_base_features(int preprocess, list clauses, int c, int v, int num_active_vars, int num_active_clauses):
    cdef dict features_dict = {}
    cdef list vcg_v_node_degrees, vcg_c_node_degrees
    cdef list vcg_v_node_degrees_norm, vcg_c_node_degrees_norm
    cdef list vg_node_degrees, vg_node_degrees_norm
    cdef list pos_neg_clause_ratios, pos_neg_clause_balance
    cdef list pos_neg_variable_ratios, pos_neg_variable_balance
    cdef int num_binary_clauses, num_ternary_clauses, num_horn_clauses
    cdef list horn_clause_variable_count
    cdef double entropy
    
    # 1-3
    features_dict["c"] = num_active_clauses
    features_dict["v"] = num_active_vars
    features_dict["clauses_vars_ratio"] = num_active_clauses / num_active_vars
    features_dict["vars_clauses_ratio"] = num_active_vars / num_active_clauses

    if not preprocess:
        pass
    else:
        c = num_active_clauses
        v = num_active_vars

    # Variable Clause Graph features
    vcg_v_node_degrees, vcg_c_node_degrees = graph_features.create_vcg(clauses, c, v)
    # variable node degrees divided by number of active clauses
    vcg_v_node_degrees_norm = [x / c for x in vcg_v_node_degrees]
    # 4-8
    write_stats(vcg_v_node_degrees_norm, "vcg_var", features_dict)
    write_entropy_discrete(vcg_v_node_degrees, c + 1, "vcg_var", features_dict)

    # clause node degrees divided by number of active variables
    vcg_c_node_degrees_norm = [x / v for x in vcg_c_node_degrees]
    # 9-13
    write_stats(vcg_c_node_degrees_norm, "vcg_clause", features_dict)
    write_entropy_discrete(vcg_c_node_degrees, v + 1, "vcg_clause", features_dict)

    # Variable graph features
    vg_node_degrees = graph_features.create_vg(clauses)
    # 14-17
    # variable node degrees divided by number of active clauses
    vg_node_degrees_norm = [x / c for x in vg_node_degrees]

    write_stats(vg_node_degrees_norm, "vg", features_dict)

    # Balance features
    pos_neg_clause_ratios, pos_neg_clause_balance, pos_neg_variable_ratios, pos_neg_variable_balance, \
    num_binary_clauses, num_ternary_clauses, num_horn_clauses, horn_clause_variable_count = \
        balance_features.compute_balance_features(clauses, c, v)
    # 18-20
    write_stats(pos_neg_clause_balance, "pnc_ratio", features_dict)
    write_entropy_continuous(pos_neg_clause_balance, "pnc_ratio", features_dict)

    # 21-25
    write_stats(pos_neg_variable_balance, "pnv_ratio", features_dict)
    write_entropy_continuous(pos_neg_variable_balance, "pnv_ratio", features_dict)

    features_dict["pnv_ratio_stdev"] = array_stats.get_stdev(pos_neg_variable_balance)
    # 26-27
    features_dict["binary_ratio"] = num_binary_clauses / c
    features_dict["ternary_ratio"] = num_ternary_clauses / c
    features_dict["ternary+"] = (num_binary_clauses + num_ternary_clauses) / c
    # 28
    features_dict["hc_fraction"] = num_horn_clauses / c
    # 29-33
    horn_clause_variable_count_norm = [x / c for x in horn_clause_variable_count]
    write_stats(horn_clause_variable_count_norm, "hc_var", features_dict)
    write_entropy_discrete(horn_clause_variable_count, c + 1, "hc_var", features_dict)

    return features_dict


# legacy
@cython.boundscheck(False)
@cython.wraparound(False)
def write_entropy(list l, str name, dict features_dict, int c, int number_of_outcomes):
    cdef double entropy
    entropy = array_stats.entropy_int_array(l, number_of_outcomes + 1)
    print("saten", entropy)
    features_dict[name + "_entropy"] = entropy

@cython.boundscheck(False)
@cython.wraparound(False)
def write_entropy_float(list l, str name, dict features_dict, int num, int buckets=100, int maxval=1):
    # scipy has an implementation for shannon entropy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html),
    # could be something to look into changing to
    cdef double entropy
    entropy = array_stats.entropy_float_array(l, num, buckets, maxval)
    print("saten", entropy)
    features_dict[name + "_entropy"] = entropy

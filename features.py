import os
import statistics
import sys
from feature_computation import parse_cnf, balance_features, graph_features, array_stats, active_features, preprocessing

'''
Main file to control extraction of features

'''


class Features:

    def __init__(self, input_cnf, preprocessing):
        self.path_to_cnf = input_cnf

        # satelite preprocessing
        self.preprocessed_path = preprocessing.satelite_preprocess(self.path_to_cnf)

        # parse the cnf file
        self.clauses, self.c, self.v = parse_cnf.parse_cnf(self.preprocessed_path)

        # computed with active features
        # These change as they are processed with dpll probing algorithms
        self.num_active_vars = 0
        self.num_active_clauses = 0
        # states and lengths of the clauses
        self.clause_states = []
        self.clause_lengths = []
        # array of the length of the number of variables, containing the number of active clauses, and binary clauses that each variable contains
        self.num_active_clauses_with_var = []
        self.num_bin_clauses_with_var = []
        # stack of indexes of the clauses that have 1 literal
        self.unit_clauses = []

        # all of the clauses that contain a positive version of this variable
        self.clauses_with_positive_var = []
        self.clauses_with_negative_var = []


        # used for dpll operations, perhaps better to keep them in a dpll class...


    def clauses_with_literal(self, literal):
        if literal > 0:
            return self.clauses_with_positive_var[literal]
        else:
            return self.clauses_with_negative_var[abs(literal)]

    def parse_active_features(self):
        self.num_active_vars, self.num_active_clauses, self.clause_states, self.clauses, self.num_bin_clauses_with_var, self.var_states =\
            active_features.get_active_features(self.clauses, self.c, self.v)




def write_stats(l, name, features_dict):
    l_mean, l_coeff, l_min, l_max = array_stats.get_stats(l)

    features_dict[name + "_mean"] = l_mean
    features_dict[name + "_coeff"] = l_coeff
    features_dict[name + "_min"] = l_min
    features_dict[name + "_max"] = l_max


def write_entropy(l, name, features_dict, c, v):
    entropy = array_stats.entropy_int_array(l, c, v+1)
    features_dict[name + "_entropy"] = entropy


def write_entropy_float(l, name, features_dict, num, buckets=100, maxval=1):
    # scipy has an implementation for shannon entropy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html),
    # could be something to look into changing to
    entropy = array_stats.entropy_float_array(l, num, buckets, maxval)
    features_dict[name + "_entropy"] = entropy


def compute_features_from_file(cnf_path="cnf_examples/basic.cnf"):
    # parse cnf, and get the features
    # store them in a dictionary
    features_dict = {}

    clauses, c, v = parse_cnf.parse_cnf(cnf_path)

    num_active_vars, num_active_clauses, clause_states, clauses = active_features.get_active_features(clauses, c, v)

    # 1-3.
    # print("c: ", c)
    # print("v: ", v)
    # print("ratio: ", c/v)
    features_dict["c"] = c
    features_dict["v"] = v
    features_dict["clauses_vars_ratio"] = c / v
    features_dict["vars_clauses_ratio"] = v / c

    # print("Active clauses: ", num_active_clauses)
    # print("Active variables: ", num_active_vars)
    # print("Active ration v/c: ", num_active_vars/num_active_clauses)

    # Variable Clause Graph features
    # 4-8
    vcg_v_node_degrees, vcg_c_node_degrees = graph_features.create_vcg(clauses, c, v)

    # variable node degrees divided by number of active clauses
    vcg_v_node_degrees_norm = [x/c for x in vcg_v_node_degrees]

    # clause node degrees divided by number of active variables
    vcg_c_node_degrees_norm = [x / v for x in vcg_c_node_degrees]

    write_stats(vcg_v_node_degrees_norm, "vcg_var", features_dict)

    write_entropy(vcg_v_node_degrees, "vcg_var", features_dict, v, c)

    # entropy needed
    # 9-13
    write_stats(vcg_c_node_degrees_norm, "vcg_clause", features_dict)
    write_entropy(vcg_c_node_degrees, "vcg_clause", features_dict, c, v)

    # 14-17
    vg_node_degrees = graph_features.create_vg(clauses)

    # variable node degrees divided by number of active clauses
    vg_node_degrees_norm = [x / c for x in vg_node_degrees]

    write_stats(vg_node_degrees_norm, "vg", features_dict)

    pos_neg_clause_ratios, pos_neg_clause_balance, pos_neg_variable_ratios, pos_neg_variable_balance,\
        num_binary_clauses, num_ternary_clauses, num_horn_clauses, horn_clause_variable_count = \
        balance_features.compute_balance_features(clauses, c, v)

    write_stats(pos_neg_clause_balance, "pnc_ratio", features_dict)
    write_entropy_float(pos_neg_clause_balance, "pnc_ratio", features_dict, c)

    write_stats(pos_neg_variable_balance, "pnv_ratio", features_dict)
    write_entropy_float(pos_neg_variable_balance, "pnv_ratio", features_dict, v)

    features_dict["pnv_ratio_stdev"] = array_stats.get_stdev(pos_neg_variable_balance)

    features_dict["binary_ratio"] = num_binary_clauses / c
    features_dict["ternary_ratio"] = num_ternary_clauses / c
    features_dict["ternary+"] = (num_binary_clauses + num_ternary_clauses) / c

    features_dict["hc_fraction"] = num_horn_clauses / c

    horn_clause_variable_count_norm = [x/c for x in horn_clause_variable_count]

    hc_var_mean, hc_var_coeff, hc_var_min, hc_var_max = array_stats.get_stats(horn_clause_variable_count_norm)
    write_entropy(horn_clause_variable_count, "hc_var", features_dict, v, c)

    features_dict["hc_var_mean"] = hc_var_mean
    features_dict["hc_var_coeff"] = hc_var_coeff
    features_dict["hc_var_min"] = hc_var_min
    features_dict["hc_var_max"] = hc_var_max

    return features_dict


if __name__ == "__main__":
    # Ideal usage - call features, with filename to calculate from, and then options on preprocessing,
    # and what features to calculate
    cnf_path = sys.argv[1]
    # preprocess_option = sys.argv[2]

    # cnf_path = "cnf_examples/basic.cnf"
    preprocessed_path = cnf_path[0:-4] + "_preprocessed.cnf"

    # n.b. satelite only works on linux, mac no longer supports 32 bit binaries...
    # satelite_preprocess(cnf_path)
    # preprocessed_path = "cnf_examples/out.cnf"
    features_dict = compute_features_from_file(preprocessed_path)

    print(features_dict)

    # static test values for local testing
    # test_labels = ["nvarsOrig", "nclausesOrig", "nvars", "nclauses", "reducedVars", "reducedClauses", "Pre-featuretime",
    #                "vars-clauses-ratio", "POSNEG-RATIO-CLAUSE-mean",
    #                "POSNEG-RATIO-CLAUSE-coeff-variation", "POSNEG-RATIO-CLAUSE-min", "POSNEG-RATIO-CLAUSE-max",
    #                "POSNEG-RATIO-CLAUSE-entropy", "VCG-CLAUSE-mean",
    #                "VCG-CLAUSE-coeff-variation", "VCG-CLAUSE-min", "VCG-CLAUSE-max", "VCG-CLAUSE-entropy", "UNARY",
    #                "BINARY+", "TRINARY+", "Basic-featuretime", "VCG-VAR-mean", "VCG-VAR-coeff-variation",
    #                "VCG-VAR-min", "VCG-VAR-max", "VCG-VAR-entropy", "POSNEG-RATIO-VAR-mean", "POSNEG-RATIO-VAR-stdev",
    #                "POSNEG-RATIO-VAR-min", "POSNEG-RATIO-VAR-max", "POSNEG-RATIO-VAR-entropy",
    #                "HORNY-VAR-mean", "HORNY-VAR-coeff-variation", "HORNY-VAR-min", "HORNY-VAR-max", "HORNY-VAR-entropy",
    #                "horn-clauses-fraction", "VG-mean", "VG-coeff-variation", "VG-min", "VG-max",
    #                "KLB-featuretime", "CG-mean", "CG-coeff-variation", "CG-min", "CG-max", "CG-entropy",
    #                "cluster-coeff-mean", "cluster-coeff-coeff-variation", "cluster-coeff-min", "cluster-coeff-max",
    #                "cluster-coeff-entropy", "CG-featuretime"]
    # test_vals = [20.000000000, 45.000000000, 15.000000000, 40.000000000, 0.333333333, 0.125000000, 0.000000000,
    #              0.375000000, 1.000000000, 0.000000000, 1.000000000, 1.000000000, -0.000000000, 0.200000000,
    #              0.577350269, 0.133333333, 0.400000000, 0.562335145, 0.000000000, 0.750000000, 0.750000000, 0.000000000,
    #              0.200000000, 0.000000000, 0.200000000, 0.200000000, -0.000000000, 0.000000000, 0.000000000,
    #              0.000000000, 0.000000000, -0.000000000, 0.100000000, 0.000000000, 0.100000000, 0.100000000,
    #              -0.000000000, 0.750000000, 0.350000000, 0.000000000, 0.350000000, 0.350000000, 0.000000000,
    #              0.262500000, 0.577350269, 0.175000000, 0.525000000, 0.562335145, 0.210227273, 0.327685288, 0.090909091,
    #              0.250000000, 0.562335145, 0.000000000]
    #
    # satzilla_features = dict(zip(test_labels, test_vals))
    #
    # print("WE ARE CHECKING")
    # print("v, c")
    # print(features_dict["v"], features_dict["c"])
    # print(satzilla_features["nvars"], satzilla_features["nclauses"])
    # print("vars clauses ratio")
    # print(features_dict["vars_clauses_ratio"])
    # print(satzilla_features["vars-clauses-ratio"])
    #
    # print("vcg clause stats")
    # print(features_dict["vcg_clause_mean"], features_dict["vcg_clause_coeff"], features_dict["vcg_clause_min"], features_dict["vcg_clause_max"], features_dict["vcg_clause_entropy"])
    # print(satzilla_features["VCG-CLAUSE-mean"], satzilla_features["VCG-CLAUSE-coeff-variation"],
    #       satzilla_features["VCG-CLAUSE-min"], satzilla_features["VCG-CLAUSE-max"], satzilla_features["VCG-CLAUSE-entropy"])
    #
    # print("vcg variable stats")
    # print(features_dict["vcg_var_mean"], features_dict["vcg_var_coeff"], features_dict["vcg_var_min"],
    #       features_dict["vcg_var_max"], features_dict["vcg_var_entropy"])
    # print(satzilla_features["VCG-VAR-mean"], satzilla_features["VCG-VAR-coeff-variation"],
    #       satzilla_features["VCG-VAR-min"], satzilla_features["VCG-VAR-max"], satzilla_features["VCG-VAR-entropy"])
    #
    # print("vg stats")
    # print(features_dict["vg_mean"], features_dict["vg_coeff"], features_dict["vg_min"],
    #       features_dict["vg_max"])
    # print(satzilla_features["VG-mean"], satzilla_features["VG-coeff-variation"],
    #       satzilla_features["VG-min"], satzilla_features["VG-max"])
    #
    # print("pos neg clauses features")
    # print(features_dict["pnc_ratio_mean"], features_dict["pnc_ratio_coeff"], features_dict["pnc_ratio_min"],
    #       features_dict["pnc_ratio_max"], features_dict["pnc_ratio_entropy"])
    # print(satzilla_features["POSNEG-RATIO-CLAUSE-mean"], satzilla_features["POSNEG-RATIO-CLAUSE-coeff-variation"],
    #       satzilla_features["POSNEG-RATIO-CLAUSE-min"], satzilla_features["POSNEG-RATIO-CLAUSE-max"], satzilla_features["POSNEG-RATIO-CLAUSE-entropy"])
    #
    # print("pos neg variable features")
    # print(features_dict["pnv_ratio_mean"], features_dict["pnv_ratio_stdev"], features_dict["pnv_ratio_min"],
    #       features_dict["pnv_ratio_max"])
    # print(satzilla_features["POSNEG-RATIO-VAR-mean"], satzilla_features["POSNEG-RATIO-VAR-stdev"],
    #       satzilla_features["POSNEG-RATIO-VAR-min"], satzilla_features["POSNEG-RATIO-VAR-max"])
    #
    # print("binary, ternary, horn_clauses")
    # print(features_dict["binary_ratio"], features_dict["ternary+"], features_dict["hc_fraction"])
    # print(satzilla_features["BINARY+"], satzilla_features["TRINARY+"], satzilla_features["horn-clauses-fraction"])
    #
    # print("horn clause variables count")
    # print(features_dict["hc_var_mean"], features_dict["hc_var_coeff"], features_dict["hc_var_min"], features_dict["hc_var_max"])
    # print(satzilla_features["HORNY-VAR-mean"], satzilla_features["HORNY-VAR-coeff-variation"], satzilla_features["HORNY-VAR-min"], satzilla_features["HORNY-VAR-max"])

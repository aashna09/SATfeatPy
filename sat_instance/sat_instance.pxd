# cython: language_level=3, boundscheck=False, wraparound=False
from feature_computation import preprocessing, parse_cnf, active_features, base_features, local_search_probing, \
    graph_features_ansotegui, graph_features_manthey_alfonso, more_graph_features
from feature_computation.dpll import DPLLProbing
from feature_computation.enums cimport VarState, ClauseState 
from sat_instance import write_to_file
from feature_computation.dpll cimport DPLLProbing
import numpy as np


cdef class SATInstance:
    """
    Class to hold the methods for generating features from a cnf. This class handles the parsing of the cnf file into
    data structures necessary to the perform feature extraction. Then the various features can be generated, and are
    stored in the features dictionary.
    """
    cdef public DPLLProbing dpll_prober
    cdef public bint verbose, preprocess, solved
    cdef public str path_to_cnf
    cdef public list clauses, unit_clauses
    cdef public int c, v, num_active_vars, num_active_clauses
    cdef public long[:] clause_lengths, num_bin_clauses_with_var, num_active_clauses_with_var
    cdef public list clause_states, clauses_with_positive_var, clauses_with_negative_var, var_states
    cdef public dict features_dict

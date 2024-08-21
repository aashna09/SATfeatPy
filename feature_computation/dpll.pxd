# cython: language_level=3

import math
import random
cimport cython
import statistics
from feature_computation.enums cimport VarState, ClauseState
from sat_instance.sat_instance cimport SATInstance
from feature_computation.stopwatch cimport Stopwatch
# from feature_computation.stopwatch cimport Stopwatch
from libc.time cimport clock, CLOCKS_PER_SEC
import numpy as np
cimport numpy as np


cdef class DPLLProbing:
    """
    So the values(states) of the variables are stored in varStates
    The clauses themselves are not changed (I assume for storage reasons)

    So assignments of the variables are stored in varstate, if they are true, false, etc,
    and then clauses are iterated through and checked if they are unassigned, so that is the next unit clause

    lengths of clauses kept track of in clause_lengths


    First check is for unit clauses -> if there are unit clauses
    Within these clauses, the unassigned variable is found
    This variable is set to the value of the literal (positive or negative)
    and the clauses that contain this variable are reduced

    reduction
    first remove all of the instances of the negative variable...
    so reduce the size of clauses with this variable,
    if the new size is 0, then we have a problem (inconsistent)

    Then satisfy the consistent clauses (clauses that contain that literal)

    Code adapted from SATzilla implementation

    """
    cdef public SATInstance sat_instance
    cdef public Stopwatch probing_stopwatch
    cdef public bint verbose
    cdef public int num_vars_to_try, num_probes, num_lob_probe
    cdef public float time_limit
    cdef public list num_reduced_clauses
    cdef public list num_reduced_vars
    cdef public list reduced_clauses
    cdef public list reduced_vars
    cdef public dict unit_props_log_nodes_dict
    cdef public dict search_space_measures_dict
    cdef public list left_subtree_size
    cdef public list branch_lengths
    cdef public list branch_probabilities
    cdef public list depths_knuth

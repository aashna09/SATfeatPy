# cython: language_level=3

from feature_computation.enums cimport VarState, ClauseState
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef VarState int_to_VarState(int state):
    if state == 1:
        return VarState.TRUE_VAL
    elif state == 2:
        return VarState.FALSE_VAL
    elif state == 3:
        return VarState.UNASSIGNED
    elif state == 4:
        return VarState.IRRELEVANT
    else:
        raise ValueError(f"Invalid state: {state}")


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ClauseState int_to_ClauseState(int state):
    if state == 1:
        return ClauseState.ACTIVE
    elif state == 2:
        return ClauseState.PASSIVE
    else:
        raise ValueError(f"Invalid state: {state}")


@cython.boundscheck(False)
@cython.wraparound(False)
def get_active_features(sat_instance, clauses, int c, int v):
    # initialize the lists that contain information on the clauses and variables
    clause_states = [ClauseState.PASSIVE] * c
    cdef long[:] num_active_clauses_with_var = np.empty(v+1, dtype=np.long)
    cdef long[:] num_bin_clauses_with_var = np.empty(v+1, dtype=np.long)
    cdef long[:] clause_lengths = np.empty(v+1, dtype=np.long)

    unit_clauses = []

    # These are used at some point...
    cdef int num_unit_clauses = 0
    cdef int num_binary_clauses = 0
    cdef int num_ternary_clauses = 0

    clauses_with_positive_var = []
    clauses_with_negative_var = []

    cdef int k
    for k in range(v + 1):
        clauses_with_positive_var.append([])
        clauses_with_negative_var.append([])

    # basically we want to remove literals if they appear twice
    # and remove the clause if it is a tautology (if variable both positive and negated appears in the clause)

    # sort the clauses. Make it a set, to remove duplicates, and iterate pairwise,
    # check if variables are negations, and if so we have a tautology

    cdef int clause_i, i, j, num_literals, literal
    for clause_i in range(c):
        clause = clauses[clause_i]
        num_literals = len(clause)
        clause.sort(key=lambda x: abs(x))

        # mark and remove redundant literals
        for i in range(num_literals - 1):
            j = i + 1
            if clause[i] == -clause[j]:
                # we have a tautology
                clauses[clause_i] = []
                break
        # remove duplicates
        clause = list(set(clause))
        clauses[clause_i] = clause

    # remove tautologies
    clauses = [cl for cl in clauses if len(cl) > 0]
    num_active_clauses = len(clauses)

    # Clause is sorted in terms of literals, and duplicates have been removed

    # Iterate through clauses again, add to binary clauses, count binary and ternary clauses
    for clause_i in range(len(clauses)):

        clause = clauses[clause_i]
        num_literals = len(clause)
        # set length of clause and
        clause_lengths[clause_i] = num_literals
        clause_states[clause_i] = ClauseState.ACTIVE

        if num_literals == 1:
            unit_clauses.append(clause_i)
            num_unit_clauses += 1
        if num_literals == 2:
            for i in range(num_literals):
                num_bin_clauses_with_var[abs(clause[i])] += 1
            num_binary_clauses += 1
        if num_literals == 3:
            num_ternary_clauses += 1

        # now go through all literals in the clause, add number of active clauses with the variable
        # and mark the clauses with positive literals, and negative literals
        for literal in clause:
            if literal < 0:
                clauses_with_negative_var[abs(literal)].append(clause_i)
            else:
                clauses_with_positive_var[abs(literal)].append(clause_i)

            num_active_clauses_with_var[abs(literal)] += 1

    # filter out the ignored clauses

    # Now remove the redundant variables
    cdef int num_active_vars = v
    cdef list var_states = [VarState.UNASSIGNED] * (v+1)

    var_states[0] = VarState.IRRELEVANT

    # check if the variable is present in any clause
    for i in range(1, v + 1):
        if num_active_clauses_with_var[i] == 0:
            var_states[i] = VarState.IRRELEVANT
            num_active_vars -= 1
        else:
            # redundant
            var_states[i] = VarState.UNASSIGNED

    # set the satinstance class variables
    sat_instance.num_active_vars = num_active_vars
    sat_instance.num_active_clauses = num_active_clauses

    sat_instance.clauses = clauses
    
    sat_instance.clause_states = [int_to_ClauseState(state) for state in clause_states]
    sat_instance.clause_lengths = clause_lengths
    sat_instance.num_active_clauses_with_var = num_active_clauses_with_var

    sat_instance.num_bin_clauses_with_var = num_bin_clauses_with_var
    sat_instance.unit_clauses = unit_clauses

    sat_instance.var_states = [int_to_VarState(state) for state in var_states]
    # all of the clauses that contain a positive version of this variable
    sat_instance.clauses_with_positive_var = clauses_with_positive_var
    sat_instance.clauses_with_negative_var = clauses_with_negative_var

    return (num_active_vars, num_active_clauses, clause_states, clauses,
            num_bin_clauses_with_var, var_states)

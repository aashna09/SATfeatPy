# cython: language_level=3
cimport cython

cdef enum VarState:
    TRUE_VAL = 1
    FALSE_VAL = 2
    UNASSIGNED = 3
    IRRELEVANT = 4


cdef enum ClauseState:
    ACTIVE = 1
    PASSIVE = 2

# cython: language_level=3

import statistics as stats
import math
import scipy.stats as sci_stats
cimport cython
import numpy as np
cimport numpy as np

"""
File to control the computation and aggregation of statistics for lists of values.
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def get_stats(list l):
    """
    Gets the four basic stats used for most features
    :param l: List to generate statistics from.
    :return: mean, co-efficient of variation, minimum and maximum.
    """
    cdef double mean = stats.mean(l)
    cdef double min_val = min(l)
    cdef double max_val = max(l)
    cdef double std = stats.pstdev(l)
    cdef double coefficient_of_variation = calc_coefficient_of_variation(mean, std)

    return mean, coefficient_of_variation, min_val, max_val


@cython.boundscheck(False)
@cython.wraparound(False)
def get_stdev(list l):
    """
    :param l: data
    :return: Population standard deviation
    """
    return stats.pstdev(l)


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_coefficient_of_variation(double mean, double std):
    """
    The coefficient of variation is a statistical measure of the relative dispersion of data points in a data series
    around the mean. https://en.wikipedia.org/wiki/Coefficient_of_variation
    :param mean:
    :param std:
    :return: Coefficient of variation
    """
    if std == 0:
        return 0
    else:
        return std / mean


@cython.boundscheck(False)
@cython.wraparound(False)
def scipy_entropy_discrete(list l, int num_outcomes):
    """
    Create a probability distribution of l, and then get the entropy of that distribution
    :param l: Data
    :param num_outcomes: The total possible number of outcomes
    :return: Entropy of l
    """
    cdef np.ndarray[np.float64_t, ndim=1] p = np.zeros(num_outcomes, dtype=np.float64)
    cdef int elem
    cdef double entropy

    for elem in l:
        p[elem] += 1

    p = p / len(l)
    entropy = sci_stats.entropy(pk=p)
    return entropy


@cython.boundscheck(False)
@cython.wraparound(False)
def scipy_entropy_continous(list l, int buckets=100):
    """
    Create a probability distribution of l, and then get the entropy of that distribution
    :param l: Data
    :return: Entropy of l
    """
    cdef np.ndarray[np.float64_t, ndim=1] p = np.zeros(buckets, dtype=np.float64)
    cdef int index, x
    cdef double entropy
    cdef double maxval = 1.0

    for x in l:
        index = math.floor(x * (buckets / maxval))
        if index >= buckets:
            index = buckets - 1
        if index < 0:
            index = 0
        p[index] += 1

    p = p / len(l)
    entropy = sci_stats.entropy(pk=p)
    return entropy

# Legacy
@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_float_array(list l, int num, int vals, double maxval):
    """
    :param l: list of values (float, should be between 0 and 1)
    :param vals: size of the bins used (normally 100)
    For now, this will be based on the implementation from SATzilla

    (posneg-ratio-clause-entropy)
    What is int num -> number of clauses/variables -> length of input array
    What is int vals -> total possible outcomes (in this case they make buckets to form a probabilty distribution function)

    writeFeature("POSNEG-RATIO-CLAUSE-entropy",array_entropy(pos_frac_in_clause,numClauses,100,1));
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    cdef np.ndarray[np.float64_t, ndim=1] p = np.zeros(vals + 1, dtype=np.float64)
    cdef double entropy = 0.0
    cdef int index, t

    for t in range(num):
        index = math.floor(l[t] * (vals / maxval))
        if index > vals:
            index = vals
        if index < 0:
            index = 0
        p[index] += 1

    for t in range(vals + 1):
        if p[t] != 0:
            pval = p[t] / num
            entropy += pval * math.log(pval)

    return -1 * entropy


@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_int_array(list l, int number_of_outcomes):
    """
    :param l: List of statistics (vcg variable/clause node degrees, )
    :param number_of_outcomes: upper bound on the maximum number of outcomes (e.g. for vcg clause node degree, it could have a maximum
    of the number of variables (the clauses contains all variables)).
    :return:
    vcg clause entropy = array_entropy(clause_array,numClauses,numActiveVars+1))
    Entropy of x  is H(X) = - sum (P(xi) * log(P(xi)))
    """
    cdef np.ndarray[np.float64_t, ndim=1] p = np.zeros(number_of_outcomes, dtype=np.float64)
    cdef double entropy = 0.0
    cdef int elem, t

    for elem in l:
        p[elem] += 1

    for t in range(number_of_outcomes):
        if p[t] != 0:
            pval = p[t] / len(l)
            entropy += pval * math.log(pval)

    return -1 * entropy

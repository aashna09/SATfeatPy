# cython: language_level=3
import cython
import math
import powerlaw
import networkx as nx
from libcpp.vector cimport vector
from libc.math cimport log, pow
import community as community_louvain
from cython.cimports import networkx as nx

cimport numpy as cnp

"""
Graph features from Structure features for SAT instances classification (Ansotegui)
------
The scale free structure (based on variable occurrences) (estimation computed by method of maximum likelihood)

exponent of the Power law distribution of variable occurrences
compute the function f_v(k), which is the number of variables that have a number of occurrences equal to k, divided by the number of variables n.
# Assuming that this function follows a power-law distribution (f_v(k) roughly = ck^-a_v), we can estimate the exponent a_v of the power law distribution that bes fits this collection of points.
This estimation is computed by the method of maximum likelihood

-Compute for all numbers k, f_v(k) = var(k)/n
-for this series, estimate the power law exponent with maximum likelihood (ask/research more about this), also the plfit package

alpha = 1 + n[sum(ln(x_i/x_min)]^-1
--------

Variable incidence graph (VIG)

Clause variable incidence graph (CVIG)

Methods for creating the VIG and CVIG should be alright

Then the fractal dimension of both the VIG and CVIG are calculated
by computing the function N(r) (Computing by burning node degrees approximation algorithm). 
This is an estimate of the minimum number of circles with radius r that can cover the graph.
This value is estimated by linear regression interpolating the points log N(r) vs log r, as N(r) ~ r^-d.

------
And the Modularity Q of the VIG is calculated
The modularity of a graph is the maximal modularity for any possible partition Q(C)  = max{Q(G,C) | C}
We can find this maximum partition with the community package (uses louvain method), and the result is the modularity of the best partition.
----  

"""


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double estimate_power_law_alpha(list clauses, int c, int v):
    """
    Estimates the power law alpha (Code adapted from Ansotegui implementation)
    :param clauses:
    :param c:
    :param v:
    :return: best fit estimated alpha
    """
    cdef list X, Y, Sylogx, Syx
    X, Y, Sylogx, Syx = variable_occurrences(clauses, c, v)
    return most_likely(X, Y, Sylogx, Syx)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple variable_occurrences(list clauses, int c, int v):
    cdef int i, count
    cdef double Sy
    cdef list variable_count = [0] * (v + 1)
    cdef list f_v_k = [0] * (c + 1)
    cdef list count_occurrences = []
    cdef list X = []
    cdef list Y = []
    cdef list Sylogx = []
    cdef list Syx = []
    
    for clause in clauses:
        for literal in clause:
            variable_count[abs(literal)] += 1

    for count in variable_count[1:]:
        f_v_k[count] += 1
    
    for count, occurrences in enumerate(f_v_k):
        if occurrences != 0:
            count_occurrences.append((count, occurrences)) 

    for count, occurrences in count_occurrences:
        Sy += occurrences
        if count > 0:
            X.append(count)  # Ensure counts are positive for log
        Y.append(0)
        Sylogx.append(0)
        Syx.append(0)

    for i in range(len(count_occurrences)-2, -1, -1):  # Adjusted the range here
        Y[i] = Y[i+1] + count_occurrences[i][1] / Sy if Sy > 0 else 0
        if X[i] > 0:  # Check to avoid math domain error
            Sylogx[i] = Sylogx[i+1] + count_occurrences[i][1] / Sy * log(X[i])
            Syx[i] = Syx[i+1] + count_occurrences[i][1] / Sy * X[i]

    return X, Y, Sylogx, Syx


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double most_likely(list X, list Y, list Sylogx, list Syx, int maxxmin=10, bint verbose=False):
    """
    Fits the data to a powerlaw
    :param X:
    :param Y:
    :param sylogx:
    :param syx:
    :param maxxmin:
    :param verbose:
    :return: The best alpha
    """
    cdef int n = len(X)
    cdef int ind, j
    cdef double xmin, alpha
    cdef double best_alpha = 0
    cdef double best_x_min_a = 0
    cdef double best_diff_a = 1
    cdef double worst_diff, worst_x
    cdef double aux
    cdef int best_ind_a = 0
    cdef double where_a = 0

    for ind in range(1, maxxmin + 1):
        if ind < n-3:
            xmin = X[ind]
            alpha = -1 - (1 / ((Sylogx[ind] / Y[ind]) - log((xmin - 0.5))))

            worst_diff = -1
            worst_x = -1

            for j in range(ind+1, n):
                aux = abs(Y[j] / Y[ind] - pow_law_c(X[j], xmin, alpha))
                if aux >= best_diff_a:
                    worst_diff = aux
                    worst_x = X[j]
                    break
                elif aux >= worst_diff:
                    worst_diff = aux
                    worst_x = X[j]

            for j in range(ind, n-1):
                if X[j] + 1 < X[j+1]:
                    aux = abs(Y[j+1] / Y[ind] - pow_law_c(X[j] + 1, xmin, alpha))
                    if aux >= best_diff_a:
                        worst_diff = aux
                        worst_x = X[j] + 1
                        # finish search of worst difference
                        break
                    elif aux >= worst_diff:
                        worst_diff = aux
                        worst_x = X[j] + 1

            if worst_diff < best_diff_a:
                best_alpha = alpha
                best_x_min_a = xmin
                best_diff_a = worst_diff
                best_ind_a = ind
                where_a = worst_x

    if verbose:
        print("alpha: ", -best_alpha)
        print("min: ", best_x_min_a)
        print("error ", best_diff_a, " in ", where_a)
        
    return -best_alpha


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double pow_law_c(double x, double xmin, double alpha):
    """
    Computes sum_{i = x} ^ {\infty} x ^ {alpha} / sum_{i = xmin} ^ {\infty} x ^ {alpha}
    or approximates it as (x / xmin) ^ (alpha + 1)
    :param x:
    :param xmin:
    :param alpha:
    :return:
    """
    cdef int max_iterations = 10000
    cdef int i
    cdef double num = 0
    cdef double den = 0
    cdef double p, p_old = -2

    assert alpha < -1
    assert xmin <= x

    if xmin < 25:
        for i in range(int(xmin), int(x)):
            den += pow(i, alpha)
        p = -1

        for i in range(int(x), int(x) + max_iterations):
            den += pow(i, alpha)
            num += pow(i, alpha)
            p_old = p
            p = num / den

            if abs(p - p_old) <= 0.00000001:
                return p
        return p

    return pow(x / xmin, alpha + 1)


cpdef double estimate_power_law_alpha_lib(list data):
    """
    Uses the powerlaw package to fit data, however this does not take into account missing data
    :param data:
    :return:
    """
    cdef list cleaned_data = [x for x in data if x > 0]
    results = powerlaw.Fit(cleaned_data)
    return results.power_law.alpha


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef create_cvig(list clauses, int c, int v):
    """
    Create a Clause Variable incidence graph
    Set of vertices is the set of variables and set of clauses,
    with weight function w(x, c) = 1/|c| if x elem c, else 0

    :param clauses:
    :param c:
    :param v:
    :return: Variable node degrees and clause node degrees
    """
    cvig = nx.Graph()
    cdef list v_nodes = [i for i in range(1, v+1)]
    cdef list c_nodes = [i for i in range(v+1, v+1+c)]

    cdef int i, k
    cdef double weight
    cdef int c_node, var_num
    cdef list abs_clause

    for i, clause in enumerate(clauses):
        abs_clause = [abs(lit) for lit in clause]
        weight = 1 / len(clause)
        c_node = c_nodes[i]

        for k, v_node in enumerate(v_nodes):
            var_num = k + 1
            if var_num in abs_clause:
                cvig.add_edge(c_node, v_node, weight=weight)

    return cvig


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef create_vig(list clauses, int c, int v):
    """
    Create a Variable incidence graph
    Set of vertices is the set of variables,
    with weight function w(x, y) = sum(1 / (|c| choose 2)) for x, y elem of c, for c elem F

    :param clauses:
    :param c:
    :param v:
    :return: Variable node degrees and clause node degrees
    """
    vig = nx.Graph()
    cdef int i, j
    cdef double weight
    cdef list clause
    cdef int v_node_i, v_node_j
    cdef dict edge_data
    cdef double edge_weight

    for clause in clauses:
        if len(clause) < 2:
            continue

        weight = 1 / math.comb(len(clause), 2)

        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                v_node_i = abs(clause[i])
                v_node_j = abs(clause[j])

                edge_data = vig.get_edge_data(v_node_i, v_node_j, default={'weight': 0})
                edge_weight = edge_data['weight'] + weight

                vig.add_edge(v_node_i, v_node_j, weight=edge_weight)

    return vig


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_modularity_q(graph):
    """
    Compute the modularity of the best partition (as estimated by the louvain method, using the python-louvain package
    :param graph:
    :return: The modularity of the graph
    """
    cdef dict partition = community_louvain.best_partition(graph)
    return community_louvain.modularity(partition, graph)


# Define the sorting key function
def get_second_item(element):
    return element[1]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list burning_by_node_degree(graph, int n):
    """
    Burning by node degree algorithm, adapted from paper pseudocode and implementation
    :param graph:
    :param n:
    :return: N(r) estimated number of circles needed to cover the graph for each circle radius r
    """
    cdef int i
    cdef list N = [0] * n
    N[1] = n
    cdef list node_degrees = []
    cdef int num_connected_components = nx.number_connected_components(graph)
    cdef int dmaxx = 16
    cdef list burned
    cdef list S
    cdef int c
    cdef int node

    for node in graph.nodes:
        degree = len(nx.edges(graph, node))
        node_degrees.append((node, degree))

    node_degrees.sort(key=get_second_item, reverse=True)
    for i in range(1, min(dmaxx + 1, len(N))):
        if N[i - 1] > num_connected_components:
            burned = [False] * (n + 1)
            burned[0] = True

            while not all(burned):
                c = highest_degree_unburned_node(node_degrees, burned)
                S = circle(c, i - 1, graph)

                for x in S:
                    burned[x] = True

                N[i] += 1

    return N


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int highest_degree_unburned_node(list node_degrees, list burned):
    """
    Get the node with the highest degree that is still unburned
    :param node_degrees:
    :param burned:
    :return:
    """
    cdef tuple node_degree
    for node_degree in node_degrees:
        node, degree = node_degree
        if not burned[node]:
            return node


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list circle(int centre, int radius, graph):
    """
    Get the nodes within the circle with centre and radius
    :param centre:
    :param radius:
    :param graph:
    :return:
    """
    subgraph = nx.generators.ego_graph(G=graph, n=centre, radius=radius)
    return list(subgraph.nodes)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple linear_regression_fit(list data):
    """
    Fit data with a linear regression and interpolation, adapted from code for ansotegui
    :param data:
    :return:
    """
    cdef list trimmed_data = [x for x in data if x > 0]
    cdef list poly_regression_X = [log(x) for x in range(1, len(trimmed_data) + 1)]
    cdef list poly_regression_Y = [log(x) for x in trimmed_data]
    cdef list exp_regression_X = [x for x in range(1, len(trimmed_data) + 1)]
    cdef list exp_regression_Y = poly_regression_Y

    cdef tuple poly = regression(poly_regression_X, poly_regression_Y)
    cdef tuple exp = regression(exp_regression_X, exp_regression_Y)

    return -poly[0], -exp[0]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple regression(list X, list Y):
    """
    Perform the regression
    :param X:
    :param Y:
    :return:
    """
    cdef double Sx = sum(X)
    cdef double Sy = sum(Y)
    cdef double Sxx = sum([x * x for x in X])
    cdef double Syy = sum([y * y for y in Y])
    cdef double Sxy = sum([x * y for (x, y) in zip(X, Y)])
    cdef double alpha, beta

    try:
        alpha = (Sx * Sy - len(X) * Sxy) / (Sx * Sx - len(X) * Sxx)
        beta = Sy / len(X) - alpha * Sx / len(X)
    except ZeroDivisionError:
        alpha = 1
        beta = 1

    return alpha, beta

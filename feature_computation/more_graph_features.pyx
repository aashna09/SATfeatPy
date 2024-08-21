# cython: language_level=3

import cython
import math
from feature_computation import array_stats
import networkx as nx
from libc.math cimport log, pow

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple create_vg(list clauses):
    """
    A variable graph (VG) has a node for each variable, and an edge between variables that occur together in at least
    one clause

    :param clauses:
    :return: The degree of each node in the variable graph
    """
    vg = nx.Graph()
    cdef:
        int i, j, k
        str v_node_i, v_node_j
        list node_degrees = []
        list weights = []
        int degree
        tuple weight_tuple

    for clause in clauses:
        k = 0
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                v_node_i = "v_" + str(abs(clause[i]))
                v_node_j = "v_" + str(abs(clause[j]))
                k += 1
                vg.add_edge(v_node_i, v_node_j, weight=pow(2, -k))

    for n in vg.nodes:
        degree = len(vg.edges(n))
        node_degrees.append(degree)
        for weight_tuple in vg.edges.data("weight", n):
            weights.append(weight_tuple[2])

    return node_degrees, weights


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple create_cg(list clauses):
    """
    A clause graph (CG) has a node for each clause, and an edge between clauses that have the same literal

    :param clauses:
    :return: The degree of each node in the variable graph and the weight for each edge
    """
    cg = nx.Graph()
    cdef:
        list c_node = []
        int i, j, degree
        str c_n
        list node_degrees = []
        list weights = []
        int weight
        int literal
        tuple weight_tuple

    for i, clause in enumerate(clauses):
        c_n = "c_" + str(i)
        c_node.append(c_n)

    for i, clause in enumerate(clauses):
        for literal in clause:
            weight = 0
            for j, clause_next in enumerate(clauses[i + 1:]):
                if literal in clause_next:
                    weight += 1
                    cg.add_edge(c_node[i], c_node[i + 1 + j], weight=weight)

    for n in cg.nodes:
        degree = len(cg.edges(n))
        for weight_tuple in cg.edges.data("weight", n):
            weights.append(weight_tuple[2])
        node_degrees.append(degree)

    return cg, node_degrees, weights


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple create_rg(list clauses):
    """
    A resolution graph (RG) has a node for each clause, and an edge between clauses if they produce a
    non-tautological resolvent

    :param clauses:
    :return: The degree of each node in the variable graph and the weight for each edge
    """
    rg = nx.Graph()
    cdef:
        list c_node = []
        int i, j, degree
        str c_n
        list node_degrees = []
        list weights = []
        int k
        double weight
        tuple weight_tuple

    for i, clause in enumerate(clauses):
        c_n = "c_" + str(i)
        c_node.append(c_n)

    for i, clause in enumerate(clauses):
        for j, clause_next in enumerate(clauses[i + 1:]):
            if len(list(set(clause) & set(clause_next))) == 1:
                k = len(set(clause)) + len(set(clause_next))
                weight = pow(2, -(k - 2))
                rg.add_edge(c_node[i], c_node[i + 1 + j], weight=weight)

    for n in rg.nodes:
        degree = len(rg.edges(n))
        for weight_tuple in rg.edges.data("weight", n):
            weights.append(weight_tuple[2])
        node_degrees.append(degree)

    return node_degrees, weights


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple create_big(list clauses):
    """
    A binary implication graph (BIG) it's a directed graph that has a node for each literal, and an edge if
    there's an implication between the literals

    :param clauses:
    :return: The degree of each node in the variable graph and the weight of each edge
    """
    big = nx.DiGraph()
    cdef:
        int i
        str v_node1, v_node2
        int a, b, degree
        list node_degrees = []
        list weights = []
        tuple weight_tuple

    for clause in clauses:
        for i in range(len(clause)):
            v_node1 = "v_" + str(abs(clause[i]))
            v_node2 = "v_" + str(-abs(clause[i]))
            big.add_node(v_node1)
            big.add_node(v_node2)

    for clause in clauses:
        if len(clause) == 2:
            a = clause[0]
            b = clause[1]
            big.add_edge('v_' + str(-a), 'v_' + str(b), weight=1)
            big.add_edge('v_' + str(-b), 'v_' + str(a), weight=1)

    for n in big.nodes:
        degree = len(big.edges(n))
        for weight_tuple in big.edges.data("weight", n):
            weights.append(weight_tuple[2])
        node_degrees.append(degree)

    return big, node_degrees, weights

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef neighbors_nodes(int l, list clauses):
    big, _, _ = create_big(clauses)
    neighbors = big.neighbors('v_' + str(l))
    return neighbors


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple create_exo_and_band(list clauses):
    """
    Create exo and band graphs based on clauses.
    """
    andg = nx.Graph()
    bandg = nx.Graph()
    exog = nx.Graph()
    cdef:
        int i, k
        str v_node1, v_node2
        list rem_clause

    for clause in clauses:
        for i in range(len(clause)):
            v_node1 = "v_" + str(abs(clause[i]))
            v_node2 = "v_" + str(-abs(clause[i]))
            andg.add_node(v_node1)
            andg.add_node(v_node2)
            bandg.add_node(v_node1)
            bandg.add_node(v_node2)
            exog.add_node(v_node1)
            exog.add_node(v_node2)

    for clause in clauses:
        if len(clause) > 2:
            exo = True
            for l0 in clause:
                k = 0
                rem_clause = clause[:]
                rem_clause.remove(l0)
                for l1 in rem_clause:
                    if l1 not in neighbors_nodes(l0, clauses):
                        exo = False
                        break
                    k += 1
                    andg.add_edge('v_' + str(l1), 'v_' + str(-l0), weight=pow(2, -k))
                if not exo:
                    break

            if not exo:
                obv_block = True
                for l0 in clause:
                    k = 0
                    rem_clause = clause[:]
                    rem_clause.remove(l0)
                    for l1 in rem_clause:
                        if len(clause) > 3:
                            obv_block = False
                            break
                        k += 1
                        bandg.add_edge('v_' + str(l1), 'v_' + str(-l0), weight=pow(2, -k))
                    if not obv_block:
                        break
            elif exo:
                for l0 in clause:
                    rem_clause = clause[:]
                    rem_clause.remove(l0)
                    for l1 in rem_clause:
                        exog.add_edge('v_' + str(l1), 'v_' + str(l0))

    return andg, bandg, exog


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple return_degrees_weights(G):
    """
    Return the degrees and weights of the nodes in the graph.
    """
    cdef:
        list node_degrees = []
        list weights = []
        int degree
        tuple weight_tuple

    for n in G.nodes:
        degree = len(G.edges(n))
        for weight_tuple in G.edges.data("weight", n):
            weights.append(weight_tuple[2])
        node_degrees.append(degree)

    return node_degrees, weights

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict recursive_weight_heuristic(int max_clause_size, list clauses, int v):
    """
    Recursive weight heuristic algorithm for clauses.
    """
    cdef:
        dict feat_dict = {}
        int i, j, clause_len, iteration, num_v, p, curr_lit, comp_ind
        double muh = 1.0
        double gamma = 5.0
        double max_double = 10e200
        double a1, a2, clause_constant, clause_value
        list this_data_pos = [0] * (v + 1)
        list this_data_neg = [0] * (v + 1)
        list this_data = [this_data_pos, this_data_neg]
        list last_data_pos = [1] * (v + 1)
        list last_data_neg = [1] * (v + 1)
        list last_data = [last_data_pos, last_data_neg]
        list all_sequences = []
        int iteration_steps
        list this_iteration_sequence
        bint found_zero

    for iteration in range(1, 4):  # 3 iterations
        iteration_steps = 0
        this_iteration_sequence = []

        for i in range(len(clauses)):
            clause = clauses[i]
            clause_len = len(clause)

            if clause_len == 1:
                continue

            if max_clause_size < clause_len:
                exponent = 0
            else:
                exponent = max_clause_size - clause_len

            try:
                a1 = math.pow(gamma, exponent)
            except OverflowError:
                a1 = float("inf")
            try:
                a2 = math.pow(muh, clause_len - 1)
            except OverflowError:
                a2 = float("inf")
                
            clause_constant = a1 / a2
            found_zero = False
            clause_value = 1.0

            for j in range(clause_len):
                curr_lit = clause[j]
                comp_ind = 0 if curr_lit < 0 else 1

                # Converting negative index values which work in python but not in cython
                if curr_lit < 0:
                    curr_lit = len(last_data[comp_ind]) + curr_lit

                # tilde is complement
                if last_data[comp_ind][curr_lit] == 0:
                    found_zero = True
                    break

                clause_value *= last_data[comp_ind][curr_lit]
                iteration_steps += 1

            if not found_zero:
                clause_value *= clause_constant

                for j in range(clause_len):
                    curr_lit = clause[j]
                    comp_ind = 0 if curr_lit < 0 else 1

                    # Converting negative index values which work in python but not in cython
                    if curr_lit < 0:
                        curr_lit = len(last_data[comp_ind]) + curr_lit

                    this_data[comp_ind][curr_lit] += clause_value / last_data[comp_ind][curr_lit]

        muh = 0.0
        for num_v in range(1, v + 1):
            for p in range(2):
                val = this_data[p][num_v]
                if val > max_double:
                    this_iteration_sequence.append(max_double)
                else:
                    this_iteration_sequence.append(val)

                muh += val

            iteration_steps += 1

        muh /= (2.0 * v)

        if muh < 1.0:
            muh = 1.0

        last_data = this_data
        this_data_pos = [0] * (v + 1)
        this_data_neg = [0] * (v + 1)
        this_data = [this_data_pos, this_data_neg]

        all_sequences.append(this_iteration_sequence)

    for i, s in enumerate(all_sequences):
        write_stats(s, "rwh_" + str(i), feat_dict)

    return feat_dict


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void write_stats(list l, str name, dict features_dict):
    """
    Compute and store statistics for a list of values.
    """
    cdef double l_mean, l_coeff, l_min, l_max

    l_mean, l_coeff, l_min, l_max = array_stats.get_stats(l)

    features_dict[name + "_mean"] = l_mean
    features_dict[name + "_coeff"] = l_coeff
    features_dict[name + "_min"] = l_min
    features_dict[name + "_max"] = l_max

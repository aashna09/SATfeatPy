# cython: language_level=3

import networkx as nx
from libc.stdlib cimport abs
from cython.parallel import prange

def create_vcg(list clauses, int c, int v):
    """
    Create VCG
    Variable-Clause Graph features
    A variable-clause graph (VCG) is a bipartite graph with a node for each variable, a node for each clause,
    and an edge between them whenever a variable occurs in a clause

    :param clauses:
    :param c:
    :param v:
    :return: Variable node degrees and clause node degrees
    """
    vcg = nx.Graph()

    # Node for each variable
    # node for each clause

    cdef int i, j, literal, degree
    cdef str c_node, v_node

    for i in range(len(clauses)):
        clause = clauses[i]
        c_node = "c_" + str(i)

        for literal in clause:
            v_node = "v_" + str(abs(literal))
            vcg.add_edge(c_node, v_node)

    cdef list v_node_degrees = []
    cdef list c_node_degrees = []
    # get node statistics
    for i in range(c):
        degree = len(vcg.edges("c_" + str(i)))
        c_node_degrees.append(degree)

    for i in range(1, v + 1):
        degree = len(vcg.edges("v_" + str(i)))
        v_node_degrees.append(degree)

    return v_node_degrees, c_node_degrees


def create_vg(list clauses):
    """
    A variable graph (VG) has a node for each variable, and an edge between variables that occur together in at least one clause

    :param clauses:
    :return: The degree of each node in the variable graph
    """
    vg = nx.Graph()

    cdef int k, i, j, clause_len, degree
    cdef str v_node_i, v_node_j

    for k in range(len(clauses)):
        clause = clauses[k]
        clause_len = len(clause)

        for i in range(clause_len):
            for j in range(i + 1, clause_len):
                v_node_i = "v_" + str(abs(clause[i]))
                v_node_j = "v_" + str(abs(clause[j]))
                vg.add_edge(v_node_i, v_node_j)

    cdef list node_degrees = []

    for n in vg.nodes:
        degree = len(vg.edges(n))
        node_degrees.append(degree)

    return node_degrees

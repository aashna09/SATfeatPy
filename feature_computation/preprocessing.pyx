# cython: language_level=3

import os
cimport cython


def satelite_preprocess(str cnf_path="cnf_examples/basic.cnf") -> str:
    # pre process using SatELite binary files
    cdef str preprocessed_path = cnf_path[0:-4] + "_preprocessed.cnf"
    cdef str satelite_command = "./SatELite/SatELite_v1.0_linux " + cnf_path + " " + preprocessed_path
    os.system(satelite_command)
    return preprocessed_path

def satelite_preprocess_tmp(str cnf_path) -> str:
    # make a temporary file
    cdef str temp_fn = os.popen("mktemp /tmp/prepro-XXXX").read().strip("\n")
    cdef str satelite_command = "./SatELite/SatELite_v1.0_linux " + cnf_path + " " + temp_fn
    os.system(satelite_command)
    return temp_fn

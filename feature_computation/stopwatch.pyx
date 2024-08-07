# cython: language_level=3

cimport cython
from libc.time cimport clock, CLOCKS_PER_SEC

cdef class Stopwatch:
    cdef double start_time

    cdef start(self):
        self.start_time = clock()

    cdef lap(self):
        cdef double c_time = clock()
        return (c_time - self.start_time) / CLOCKS_PER_SEC

# cython: language_level=3

cimport cython
from libc.time cimport clock, CLOCKS_PER_SEC

cdef class Stopwatch:
    def start(self):
        self.start_time = clock()

    def lap(self):
        cdef double c_time = clock()
        return (c_time - self.start_time) / CLOCKS_PER_SEC

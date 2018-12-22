# cython: infer_types=True
cimport cython
import numpy as np
import scipy.linalg.blas

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)

def protocol_to_base10_int(long [:] protocol, int x1, int x2):
        cdef int value, p=2, b, l, i, x2m1
        value = protocol[x2-1]
        x2m1 = x2-1
        l = x2-x1
        for i in range(1, l):
                value = value + p*protocol[x2m1-i]
                p*=2
        return value
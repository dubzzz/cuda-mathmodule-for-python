import numpy as np
cimport numpy as np

cdef extern from "objects/Vector.hpp":
    cdef cppclass Vector:
        Vector(unsigned int size)
        Vector(double *h_v, unsigned int size)
        void memsetZero()
        np.ndarray toNumPy()
        unsigned int getSize()

cdef class PyVector:
    cdef Vector *thisptr
    def __cinit__(self, unsigned int size):
        self.thisptr = new Vector(size)
    def toNumPy(self):
        return self.thisptr.toNumPy()


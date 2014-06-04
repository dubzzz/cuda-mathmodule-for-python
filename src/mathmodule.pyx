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
    
    def __cinit__(self, *args):
        """
        Init a Vector
        1. From of a given size
           WARNING: values in the array are not defined (memory-dependant)
        2. From an existing NumPy array
        """
        
        cdef np.ndarray ndarray
        types = [type(arg) for arg in args]
        if types == [int]:
            self.thisptr = new Vector(args[0])
        elif types == [np.ndarray]:
            ndarray = args[0]
            if ndarray.ndim == 1:
                self.thisptr = new Vector(<double*> ndarray.data, ndarray.shape[0])
            elif ndarray.ndim == 2 and ndarray.shape[1] == 1:
                self.thisptr = new Vector(<double*> ndarray.data, ndarray.shape[0])
            elif ndarray.ndim == 3 and ndarray.shape[1] == 1 and ndarray.shape[2] == 1:
                self.thisptr = new Vector(<double*> ndarray.data, ndarray.shape[0])
            else:
                raise ValueError("PyVector::__cinit__\tMulti-dimensional arrays cannot be converted to PyVector objects")
        else:
            raise TypeError("PyVector::__cinit__\tUnknown method signature")

    def __dealloc__(self):
        del self.thisptr
    
    def memsetZero(self):
        """ Reset array with zeros """
        self.thisptr.memsetZero()
    
    def size(self):
        """ Size of the array """
        return self.thisptr.getSize()
    
    def toNumPy(self):
        """ NumPy equivalent of CUDA Vector """
        return self.thisptr.toNumPy()


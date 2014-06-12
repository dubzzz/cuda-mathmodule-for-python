#!/usr/bin/python
import numpy as np
import sys
import unittest

sys.path.append("../src/")
sys.path.append("tests/../src/")
import mathmodule as mm

height = 4096
width = 4096
error_max = 1e-8

def check_equals_double(d1, d2):
    return abs(d1-d2) <= error_max

def check_equals_ndarray(array1, array2):
    if array1.ndim == 1:
        for i in range(array1.shape[0]):
            if abs(array1[i]-array2[i]) > error_max:
                return False
        return True
    elif array1.ndim == 2:
        for i in range(array1.shape[0]):
            for j in range(array1.shape[1]):
                if abs(array1[i,j]-array2[i,j]) > error_max:
                    return False
    elif array1.ndim == 3:
        for i in range(array1.shape[0]):
            for j in range(array1.shape[1]):
                for k in range(array1.shape[2]):
                    if abs(array1[i,j,k]-array2[i,j,k]) > error_max:
                        return False
        return True
    return False

class TestPyVector(unittest.TestCase):
    def test_toNumPy(self):
        a = np.random.random(width)
        va = mm.PyVector(a)
        self.assertTrue(check_equals_ndarray(va.toNumPy(), a))
    
    def test_iadd(self):
        a = np.random.random(width)
        b = np.random.random(width)
        va = mm.PyVector(a)
        vb = mm.PyVector(b)
        va += vb
        self.assertTrue(check_equals_ndarray(va.toNumPy(), a+b))
    
    def test_add(self):
        a = np.random.random(width)
        b = np.random.random(width)
        va = mm.PyVector(a)
        vb = mm.PyVector(b)
        vc = va + vb
        self.assertTrue(check_equals_ndarray(vc.toNumPy(), a+b))
    
    def test_idot(self):
        a = np.random.random(width)
        b = np.random.random(width)
        va = mm.PyVector(a)
        vb = mm.PyVector(b)
        self.assertTrue(check_equals_double(va.dot(vb), a.dot(b)))
    
    def test_dot(self):
        a = np.random.random(width)
        b = np.random.random(width)
        va = mm.PyVector(a)
        vb = mm.PyVector(b)
        self.assertTrue(check_equals_double(mm.dot(va, vb), a.dot(b)))

if __name__ == '__main__':
    unittest.main()


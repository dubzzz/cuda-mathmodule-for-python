#!/usr/bin/python
import sys
import timeit as ti

sys.path.append("../src/")
sys.path.append("tests/../src/")

num_tests = 100

setup = """
import numpy as np
import mathmodule as mm

height = 40960
width = 40960

a = np.random.random(width)
b = np.random.random(width)
va = mm.PyVector(a)
vb = mm.PyVector(b)
"""

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        try:
            num_tests = int(sys.argv[1])
        except ValueError:
            pass
    
    print "PyVector::__iadd__"
    print "> cuda: ", ti.timeit(stmt='va += vb', number=num_tests, setup=setup)
    print "> numpy:", ti.timeit(stmt='a += b', number=num_tests, setup=setup)

    print "PyVector::__add__"
    print "> cuda: ", ti.timeit(stmt='va + vb', number=num_tests, setup=setup)
    print "> numpy:", ti.timeit(stmt='a + b', number=num_tests, setup=setup)


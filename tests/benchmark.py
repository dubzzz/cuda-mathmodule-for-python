#!/usr/bin/python
import sys
import timeit as ti

sys.path.append("../src/")
sys.path.append("tests/../src/")

num_tests = 100
default_size = 4096

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        try:
            num_tests = int(sys.argv[1])
        except ValueError:
            pass
    
    try:
        sizes = [int(s) for s in sys.argv[2:]]
    except ValueError:
        sizes = list()
    
    if len(sizes) == 0:
        sizes.append(default_size)
    
    for size in sizes:
        setup = """
import numpy as np
import mathmodule as mm

height = %d
width = %d

a = np.random.random(width)
b = np.random.random(width)
va = mm.PyVector(a)
vb = mm.PyVector(b)
""" % (size, size)
        
        print "\nBENCHMARK -- #tests=", num_tests, "- size=", size, "\n"
        
        print "PyVector::__iadd__"
        print "> cuda: ", ti.timeit(stmt='va += vb', number=num_tests, setup=setup)
        print "> numpy:", ti.timeit(stmt='a += b', number=num_tests, setup=setup)

        print "PyVector::__add__"
        print "> cuda: ", ti.timeit(stmt='va + vb', number=num_tests, setup=setup)
        print "> numpy:", ti.timeit(stmt='a + b', number=num_tests, setup=setup)

        print "PyVector::dot"
        print "> cuda: ", ti.timeit(stmt='va.dot(vb)', number=num_tests, setup=setup)
        print "> numpy:", ti.timeit(stmt='a.dot(b)', number=num_tests, setup=setup)

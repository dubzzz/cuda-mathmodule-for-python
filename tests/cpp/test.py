#!/usr/bin/python
import sys
import timeit as ti

sys.path.append("../../src/")
import mathmodule as mm

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "not enough arguments"
        exit(0)
    
    function = sys.argv[1]
    size = int(sys.argv[2])
    module = sys.argv[3]
    
    setup = """import numpy as np
import mathmodule as mm
width = %d
a = np.random.random(width)
b = np.random.random(width)
va = mm.PyVector(a)
vb = mm.PyVector(b)
""" % (size)
    
    if function == "add":
        if module == "cuda":
            print 1000*ti.timeit(stmt='va+vb', number=1, setup=setup)
        else:
            print 1000*ti.timeit(stmt='a+b', number=1, setup=setup)
    elif function == "iadd":
        if module == "cuda":
            print 1000*ti.timeit(stmt='va+=vb', number=1, setup=setup)
        else:
            print 1000*ti.timeit(stmt='a+=b', number=1, setup=setup)
    else:
        print "unknown function"

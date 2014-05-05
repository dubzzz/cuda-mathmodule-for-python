#!/usr/bin/python
import numpy
import time
import sys

sys.path.append("../bin/")
import mathmodule

a = numpy.random.random(1024)
b = numpy.random.random(1024)

start = time.time()
print "MathModule.dot: ", mathmodule.dot(a, b)
end = time.time()
print "> time: ", 1000*(end - start), "ms"

start = time.time()
print "NumPy.dot: ", numpy.dot(a, b)
end = time.time()
print "> time: ", 1000*(end - start), "ms"


#!/usr/bin/python
import numpy
import time
import sys

sys.path.append("../bin/")
import mathmodule

height = 4096
width = 4096

ma = numpy.random.random([height, width])
a = numpy.random.random(width)
b = numpy.random.random(width)

print ma
print a
print b

print "\nDOT PRODUCT:\n"

start = time.time()
print "MathModule:", mathmodule.dot(a, b)
end = time.time()
print "> time: ", 1000*(end - start), "ms"

start = time.time()
print "NumPy:     ", numpy.dot(a, b)
end = time.time()
print "> time: ", 1000*(end - start), "ms"

print "\nMATRIX x VECTOR PRODUCT:\n"

start = time.time()
print "MathModule:", mathmodule.product(ma, a)
end = time.time()
print "> time: ", 1000*(end - start), "ms"

start = time.time()
print "NumPy:     ", numpy.asmatrix(ma) * numpy.asmatrix(a).transpose()
end = time.time()
print "> time: ", 1000*(end - start), "ms"



#ifndef __PYTHONCHECKS_HPP__
#define __PYTHONCHECKS_HPP__

#include <Python.h> // always first
#include <arrayobject.h>

bool not_doublevector(PyArrayObject *vec);
bool not_doublematrix(PyArrayObject *mat);

#endif


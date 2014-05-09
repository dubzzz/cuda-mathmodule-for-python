#include "PythonChecks.hpp"

bool not_doublevector(PyArrayObject *vec)
{
	if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)
	{
		PyErr_SetString(PyExc_ValueError, "In not_doublevector: array must be of type Float and 1 dimensional (n).");
		return true;
	}
	return false;
}

bool not_doublematrix(PyArrayObject *mat)
{
	if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2)
	{
		PyErr_SetString(PyExc_ValueError, "In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
		return true;
	}
	return false;
}


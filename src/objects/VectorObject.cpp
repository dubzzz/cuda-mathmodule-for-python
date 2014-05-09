#include "VectorObject.hpp"
#include "../preproc.hpp"

#define isVectorPtr(X) (((VectorObject*) X)->ptr_vector != 0)
#define getVectorPtr(X) ((Vector*) ((VectorObject*) X)->ptr_vector)
#define setVectorPtr(X, Y) {((VectorObject*) X)->ptr_vector = (unsigned long long) Y;}
#define resetVectorPtr(X) setVectorPtr(X, 0)

void init_vectorobject()
{__LOG__
	import_array();
}

PyObject *Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{__LOG__
	VectorObject *self;
	self = (VectorObject*) type->tp_alloc(type, 0);
	if (self == NULL) return NULL;
	resetVectorPtr(self);
	return (PyObject*) self;
}

int Vector_init(VectorObject *self, PyObject *args, PyObject *kwds)
{__LOG__
	PyArrayObject *vect;
	
	if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &vect))
		return -1;
	
	if (vect == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_Vector_init: array vect must be defined");
		return -1;
	}
	if (not_doublevector(vect)) return -1;
	
	setVectorPtr(self, new Vector((double*) vect->data, vect->dimensions[0]));
	return 0;
}

void Vector_dealloc(VectorObject *self)
{__LOG__
	if (isVectorPtr(self))
	{
		delete getVectorPtr(self);
		resetVectorPtr(self);
	}
	
	self->ob_type->tp_free((PyObject*) self);
}

PyObject *Vector_toNumPy(VectorObject *self)
{__LOG__
	return PyArray_Return(getVectorPtr(self)->toNumPy());
}


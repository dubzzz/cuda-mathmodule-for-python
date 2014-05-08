#include "VectorObject.hpp"

void init_vectorobject()
{
	import_array();
}

PyObject *Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	VectorObject *self;
	self = (VectorObject*) type->tp_alloc(type, 0);
	if (self == NULL) return NULL;
	return (PyObject*) self;
}

int Vector_init(VectorObject *self, PyObject *args, PyObject *kwds)
{
	PyArrayObject *vect;
	
	if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &vect))
		return -1;
	
	if (vect == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_Vector_init: array vect must be defined");
		return -1;
	}
	if (not_doublevector(vect)) return -1;
	
	self->v = new Vector((double*) vect->data, vect->dimensions[0]);
	return 0;
}

void Vector_dealloc(VectorObject *self)
{
	delete self->v;
}

PyObject *Vector_toNumPy(VectorObject *self)
{
	return PyArray_Return(self->v->toNumPy());
}


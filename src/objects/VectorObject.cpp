#include "VectorObject.hpp"
#include "../preproc.hpp"

#include "kernels/add.hpp"

#define isVectorPtr(X) (((VectorObject*) X)->ptr_vector != 0)
#define getVectorPtr(X) ((Vector*) ((VectorObject*) X)->ptr_vector)
#define setVectorPtr(X, Y) {((VectorObject*) X)->ptr_vector = (unsigned long long) Y;}
#define resetVectorPtr(X) setVectorPtr(X, 0)

void init_vectorobject()
{__LOG__
	import_array();
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
{__LOG(self->ob_refcnt)
	if (isVectorPtr(self) && self->ob_refcnt == 0)
	{
		delete getVectorPtr(self);
		resetVectorPtr(self);
	}
	
	self->ob_type->tp_free((PyObject*) self);
}

PyObject *Vector_add(PyObject *a, PyObject *b)
{__LOG__
	VectorObject *voa, *vob, *voc;
	if (! isVector(a) || ! isVector(b))
	{
		PyErr_SetString(PyExc_TypeError, "In mathmodule_Vector_add: a and b must be defined as Vector instances");
		return NULL;
	}
	voa = (VectorObject*) a;
	vob = (VectorObject*) b;
	voc = (VectorObject*) PyObject_New(VectorObject, &VectorType);
	if (voc == NULL) return NULL;
	
	setVectorPtr(voc, new Vector(getVectorPtr(voa)->getSize()));
	add(*getVectorPtr(voa), *getVectorPtr(vob), *getVectorPtr(voc));
	return (PyObject*) voc;
}

PyObject *Vector_iadd(PyObject *self, PyObject *b)
{__LOG__
	VectorObject *voa, *vob, *voc;
	if (! isVector(self) || ! isVector(b))
	{
		PyErr_SetString(PyExc_TypeError, "In mathmodule_Vector_iadd: a and b must be defined as Vector instances");
		return NULL;
	}
	voa = (VectorObject*) self;
	vob = (VectorObject*) b;
	
	add(*getVectorPtr(voa), *getVectorPtr(vob), *getVectorPtr(voa));
	return Py_BuildValue("O", (PyObject*) voa);
}

PyObject *Vector_toNumPy(VectorObject *self)
{__LOG__
	return PyArray_Return(getVectorPtr(self)->toNumPy());
}


#ifndef __VECTOROBJECT_HPP__
#define __VECTOROBJECT_HPP__

#include <Python.h>
#include <arrayobject.h>

#include "../checks/PythonChecks.hpp"
#include "Vector.hpp"

typedef struct
{
	PyObject_HEAD
	Vector *v;
} VectorObject;

/*
	Call this method before calling anything else
	
	if another method which requires NumPy is called without having called this method,
	the program will return a segmentation fault
*/
void init_vectorobject();

PyObject *Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int Vector_init(VectorObject *self, PyObject *args, PyObject *kwds);
void Vector_dealloc(VectorObject *self);
PyObject *Vector_numpy(VectorObject *self);

static PyMethodDef Vector_methods[] =
{
	{"numpy", (PyCFunction)Vector_numpy, METH_NOARGS, "Return the numpy equivalent of the object"},
	{NULL} /* Sentinel */
};

static PyTypeObject VectorType =
{
	PyObject_HEAD_INIT(NULL)
	0,			/*ob_size*/
	"mathmodule.Vector",	/*tp_name*/
	sizeof(VectorObject),	/*tp_basicsize*/
	0,			/*tp_itemsize*/
	(destructor) Vector_dealloc,	/*tp_dealloc*/
	0,			/*tp_print*/
	0,			/*tp_getattr*/
	0,			/*tp_setattr*/
	0,			/*tp_compare*/
	0,			/*tp_repr*/
	0,			/*tp_as_number*/
	0,			/*tp_as_sequence*/
	0,			/*tp_as_mapping*/
	0,			/*tp_hash */
	0,			/*tp_call*/
	0,			/*tp_str*/
	0,			/*tp_getattro*/
	0,			/*tp_setattro*/
	0,			/*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,	/*tp_flags*/
	"Vector objects",	/* tp_doc */
	0,			/* tp_traverse */
	0,			/* tp_clear */
	0,			/* tp_richcompare */
	0,			/* tp_weaklistoffset */
	0,			/* tp_iter */
	0,			/* tp_iternext */
	Vector_methods,		/* tp_methods */
	0,			/* tp_members */
	0,			/* tp_getset */
	0,			/* tp_base */
	0,			/* tp_dict */
	0,			/* tp_descr_get */
	0,			/* tp_descr_set */
	0,			/* tp_dictoffset */
	(initproc) Vector_init,	/* tp_init */
	0,			/* tp_alloc */
	Vector_new,		/* tp_new */
};

#endif


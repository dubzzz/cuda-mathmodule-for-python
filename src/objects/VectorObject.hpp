#ifndef __VECTOROBJECT_HPP__
#define __VECTOROBJECT_HPP__

#include <Python.h>
#include <structmember.h>
#include <arrayobject.h>
#include <typeinfo>

#include "../checks/PythonChecks.hpp"
#include "Vector.hpp"

#define isVector(X) (true)
//((X)->ob_type == &VectorType)

typedef struct
{
	PyObject_HEAD
	unsigned long long ptr_vector;
} VectorObject;

/*
	Call this method before calling anything else
	
	if another method which requires NumPy is called without having called this method,
	the program will return a segmentation fault
*/
void init_vectorobject();

int Vector_init(VectorObject *self, PyObject *args, PyObject *kwds);
void Vector_dealloc(VectorObject *self);

PyObject *Vector_add(PyObject *a, PyObject *b);
PyObject *Vector_iadd(PyObject *self, PyObject *b);

PyObject *Vector_toNumPy(VectorObject *self);

static PyMemberDef Vector_members[] =
{
	{"ptr_vector", T_ULONGLONG, offsetof(VectorObject, ptr_vector), READONLY, "Pointer to Vector GPU-size object"},
	{NULL} /* Sentinel */
};

static PyMethodDef Vector_methods[] =
{
	{"toNumPy", (PyCFunction)Vector_toNumPy, METH_NOARGS, "Return the NumPy equivalent of the object"},
	{NULL} /* Sentinel */
};

static PyNumberMethods Vector_as_number = {
	Vector_add,	/* nb_add */
	0,	/* nb_subtract */
	0,	/* nb_multiply */
	0,	/* nb_divide */
	0,	/* nb_remainder */
	0,	/* nb_divmod */
	0,	/* nb_power */
	0,	/* nb_negative */
	0,	/* nb_positive */
	0,	/* nb_absolute */
	0,	/* nb_nonzero */
	0,	/* nb_invert */
	0,	/* nb_lshift */
	0,	/* nb_rshift */
	0,	/* nb_and */
	0,	/* nb_xor */
	0,	/* nb_or */
	0,	/* nb_coerce */
	0,	/* nb_int */
	0,	/* nb_long */
	0,	/* nb_float */
	0,	/* nb_oct */
	0,	/* nb_hex */
	Vector_iadd,	/* nb_inplace_add */
	0,	/* nb_inplace_subtract */
	0,	/* nb_inplace_multiply */
	0,	/* nb_inplace_divide */
	0,	/* nb_inplace_remainder */
	0,	/* nb_inplace_power */
	0,	/* nb_inplace_lshift */
	0,	/* nb_inplace_rshift */
	0,	/* nb_inplace_and */
	0,	/* nb_inplace_xor */
	0,	/* nb_inplace_or */
	0,	/* nb_floor_divide */
	0,	/* nb_true_divide */
	0,	/* nb_inplace_floor_divide */
	0,	/* nb_inplace_true_divide */
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
	&Vector_as_number,	/*tp_as_number*/
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
	Vector_members,		/* tp_members */
	0,			/* tp_getset */
	0,			/* tp_base */
	0,			/* tp_dict */
	0,			/* tp_descr_get */
	0,			/* tp_descr_set */
	0,			/* tp_dictoffset */
	(initproc) Vector_init,	/* tp_init */
	0,			/* tp_alloc */
	0,			/* tp_new */
};

#endif


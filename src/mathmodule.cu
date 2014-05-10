#include <Python.h> // always first
#include <arrayobject.h>

#include "mathmodule.hpp"

#include "checks/CudaChecks.hpp"
#include "checks/PythonChecks.hpp"

#include "kernels/dot.hpp"
#include "kernels/product.hpp"

#include "objects/Matrix.hpp"
#include "objects/Vector.hpp"
#include "objects/VectorObject.hpp"

#include "preproc.hpp"

static PyMethodDef MathModuleMethods[] =
{
	{"dot", mathmodule_dot, METH_VARARGS, "Compute the value of the dot product of two NumPy arrays"},
	{"product", mathmodule_product, METH_VARARGS, "Compute the product of Matrix x Vector NumPy arrays"},
	{NULL, NULL, 0, NULL}, //Sentinel: end of the structure
};

__global__ void init_kernel() {}

PyMODINIT_FUNC initmathmodule(void)
{__LOG__
	/* Init module */
	
	PyObject* m;
	m = Py_InitModule3("mathmodule", MathModuleMethods, "Py mathematics module using CUDA");
	if (m == NULL)
		return;
	
	import_array(); // for NumPy
	init_matrix();
	init_vector();
	init_vectorobject();
	
	/* Add objects: Vector */
	
	VectorType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&VectorType) < 0)
		return;
	
	Py_INCREF(&VectorType);
	PyModule_AddObject(m, "Vector", (PyObject *)&VectorType);
	
	/* Launch first kernel */
	
	// first call to a CUDA function will normally takes more time than other calls
	// during this call, the module has to find the GPU to use, its version..
	init_kernel<<<1,1>>>();
}

static PyObject *mathmodule_dot(PyObject *self, PyObject *args)
{__LOG__
	PyArrayObject *vect1, *vect2;
	double *h_cvect1, *h_cvect2;
	
	// if an error is detected in the argument list
	if (! PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &vect1, &PyArray_Type, &vect2))
		return NULL;
	
	if (vect1 == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_dot: array vect1 must be defined");
		return NULL;
	}
	if (vect2 == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_dot: array vect2 must be defined");
		return NULL;
	}
	
	// Check that objects are 'double' type and vectors
	if (not_doublevector(vect1)) return NULL;
	if (not_doublevector(vect2)) return NULL;
	
	// Change contiguous arrays into C *arrays
	h_cvect1 = (double*) vect1->data;
	h_cvect2 = (double*) vect2->data;
	
	const unsigned long int n = vect1->dimensions[0];
	Vector d_vect1(h_cvect1, n);
	Vector d_vect2(h_cvect2, n);
	
	// Return dot-product result
	return Py_BuildValue("d", dot(d_vect1, d_vect2));
}

static PyObject *mathmodule_product(PyObject *self, PyObject *args)
{__LOG__
	PyArrayObject *mat, *vect;
	double *h_cmat, *h_cvect;
	
	// if an error is detected in the argument list
	if (! PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &mat, &PyArray_Type, &vect))
		return NULL;
	
	if (mat == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_product: matrix mat must be defined");
		return NULL;
	}
	if (vect == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_product: array vect must be defined");
		return NULL;
	}
	
	// Check that objects are 'double' type and matrix/vector
	if (not_doublematrix(mat)) return NULL;
	if (not_doublevector(vect)) return NULL;
	
	// Change contiguous arrays into C *arrays
	h_cmat = (double*) mat->data;
	h_cvect = (double*) vect->data;
	
	Matrix d_mat(h_cmat, mat->dimensions[0], mat->dimensions[1]);
	Vector d_vect(h_cvect, vect->dimensions[0]);
	
	Vector *d_vect_product = product(d_mat, d_vect);
	if (d_vect_product == NULL) return NULL;
	PyArrayObject *numpy_array = d_vect_product->toNumPy();
	delete d_vect_product;
	
	return PyArray_Return(numpy_array);
}

int main(int argc, char *argv[])
{__LOG__
	// Pass argv[0] to the Python interpreter
	Py_SetProgramName(argv[0]);
	
	// Initialize the Python interpreter. Required.
    	Py_Initialize();
	
	// Add a static module
	initmathmodule();
}


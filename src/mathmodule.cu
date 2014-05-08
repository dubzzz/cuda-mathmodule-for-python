#include <Python.h> // always first
#include <arrayobject.h>

#include "errors.cuh"

#include "Matrix.cuh"
#include "Vector.cuh"
#include "mathmodule.hpp"

//#define _DEBUG
#ifdef _DEBUG
	#define __start() cudaEvent_t start, stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start, 0);
	#define __stop() cudaEventRecord(stop, 0);cudaEventSynchronize(stop);float time;cudaEventElapsedTime(&time, start, stop);std::cout << "CUDA Kernel: " << time << "ms" << std::endl;
#else
	#define __start() 
	#define __stop()
#endif

#define MAX_THREADS 1024

static PyMethodDef MathModuleMethods[] =
{
	{"dot", mathmodule_dot, METH_VARARGS, "Compute the value of the dot product of two NumPy arrays"},
	{"product", mathmodule_product, METH_VARARGS, "Compute the product of Matrix x Vector NumPy arrays"},
	{NULL, NULL, 0, NULL}, //Sentinel: end of the structure
};

typedef struct
{
	PyObject_HEAD
	Vector *v;
} VectorObject;

static PyObject *Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	VectorObject *self;
	self = (VectorObject*) type->tp_alloc(type, 0);
	if (self == NULL) return NULL;
	return (PyObject*) self;
}

static int Vector_init(VectorObject *self, PyObject *args, PyObject *kwds)
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

static void Vector_dealloc(VectorObject *self)
{
	delete self->v;
}

static PyObject *Vector_numpy(VectorObject *self)
{
	return PyArray_Return(self->v->toNumPy());
}

static PyMethodDef Vector_methods[] =
{
	{"numpy", (PyCFunction)Vector_numpy, METH_NOARGS, "Return the numpy equivalent of the object"},
	{NULL}  /* Sentinel */
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

__global__ void init_kernel() {}

PyMODINIT_FUNC initmathmodule(void)
{
	/* Init module */
	
	PyObject* m;
	m = Py_InitModule3("mathmodule", MathModuleMethods, "Py mathematics module using CUDA");
	if (m == NULL)
		return;
	
	import_array(); // for NumPy
	
	/* Add objects: Vector */
	
	if (PyType_Ready(&VectorType) < 0)
		return;
	
	Py_INCREF(&VectorType);
	PyModule_AddObject(m, "Vector", (PyObject *)&VectorType);
	
	/* Launch first kernel */
	
	// first call to a CUDA function will normally takes more time than other calls
	// during this call, the module has to find the GPU to use, its version..
	init_kernel<<<1,1>>>();
}

__device__ double atomicAdd(double* address, double val) // http://stackoverflow.com/questions/16882253/cuda-atomicadd-produces-wrong-result
{
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	}
	while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void dot_kernel(const Vector d_vect1, const Vector d_vect2, double *dot_result)
{
	/*
		Each block has to evaluate its the dot of
		blockDim.x elements by blockDim.x
		
		Once it is done it adds the value to dot_result
	*/
	__shared__ double cache[MAX_THREADS];
	
	int cacheIdx = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	cache[cacheIdx] = i >= d_vect1.getSize() ? 0. : d_vect1[i] * d_vect2[i];
	__syncthreads();
	
	if (i >= d_vect1.getSize())
		return;
	
	int padding = blockDim.x/2;
	while (padding != 0)
	{
		if (cacheIdx < padding)
			cache[cacheIdx] += cache[cacheIdx + padding];
		
		__syncthreads();
		padding /= 2;
	}
	
	if (cacheIdx == 0)
		atomicAdd(&dot_result[0], cache[0]);
}

static PyObject *mathmodule_dot(PyObject *self, PyObject *args)
{
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
	
	// Check dimensions
	const unsigned long int n = vect1->dimensions[0];
	if (vect2->dimensions[0] != n)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_dot: arrays vect1 and vect2 must have the same dimensions");
		return NULL;
	}
	
	// Change contiguous arrays into C *arrays
	h_cvect1 = (double*) vect1->data;
	h_cvect2 = (double*) vect2->data;
	
	// CUDA dot-product
	Vector d_vect1(h_cvect1, n);
	Vector d_vect2(h_cvect2, n);
	double *d_dot_result;
	cudaMalloc(&d_dot_result, sizeof(double));
	cudaMemset(d_dot_result, 0, sizeof(double));
	__start()
	dot_kernel<<<(n + MAX_THREADS -1)/MAX_THREADS, MAX_THREADS>>>(d_vect1, d_vect2, d_dot_result);
	cudaThreadSynchronize(); // block until the device is finished
	__stop()
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		cudaFree(d_dot_result);
		PyErr_SetString(PyExc_RuntimeError, "In mathmodule_dot: CUDA failed");
		return NULL;
	}
	
	double dot_result;
	cudaMemcpy(&dot_result, d_dot_result, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_dot_result);
	
	// Return dot-product result
	return Py_BuildValue("d", dot_result);
}

__global__ void product_kernel(const Matrix d_mat, const Vector d_vect, Vector d_vect_result)
{
	/*
		Each block has to evaluate the product of its line elements
		by the corresponding vector elements
		
		A block can only be responsible for one line
		If the line is to large for one block, the work is split among several blocks
	*/
	
	__shared__ double cache[MAX_THREADS];
	
	int cacheIdx = threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	cache[cacheIdx] = (i >= d_mat.getHeight() || j >= d_mat.getWidth()) ? 0. : d_mat.get(i, j) * d_vect[j];
	__syncthreads();
	
	if (i >= d_mat.getHeight() || j >= d_mat.getWidth())
		return;
	
	int padding = blockDim.y/2;
	while (padding != 0)
	{
		if (cacheIdx < padding)
			cache[cacheIdx] += cache[cacheIdx + padding];
		
		__syncthreads();
		padding /= 2;
	}
	
	if (cacheIdx == 0)
		atomicAdd(&d_vect_result[i], cache[0]);
}

static PyObject *mathmodule_product(PyObject *self, PyObject *args)
{
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
	
	// Check dimensions
	if (mat->dimensions[1] != vect->dimensions[0])
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_product: dim1 of mat must be equal to dim0 of vect");
		return NULL;
	}
	
	// Change contiguous arrays into C *arrays
	h_cmat = (double*) mat->data;
	h_cvect = (double*) vect->data;
	
	// CUDA product
	Matrix d_mat(h_cmat, mat->dimensions[0], mat->dimensions[1]);
	Vector d_vect(h_cvect, vect->dimensions[0]);
	Vector d_vect_result(mat->dimensions[0]);
	d_vect_result.memsetZero();
	__start()
	const dim3 num_threads(1, MAX_THREADS, 1);
	const dim3 num_blocks((int)mat->dimensions[0], ((int)mat->dimensions[1] + MAX_THREADS -1)/MAX_THREADS, 1);
	product_kernel<<<num_blocks, num_threads>>>(d_mat, d_vect, d_vect_result);
	cudaThreadSynchronize(); // block until the device is finished
	__stop()
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		PyErr_SetString(PyExc_RuntimeError, "In mathmodule_product: CUDA failed");
		return NULL;
	}
	
	return PyArray_Return(d_vect_result.toNumPy());
}

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

int main(int argc, char *argv[])
{
	// Pass argv[0] to the Python interpreter
	Py_SetProgramName(argv[0]);
	
	// Initialize the Python interpreter. Required.
    	Py_Initialize();
	
	// Add a static module
	initmathmodule();
}


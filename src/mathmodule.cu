#include <Python.h> // always first
#include <arrayobject.h>

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
	{NULL, NULL, 0, NULL}, //Sentinel: end of the structure
};

__global__ void init_kernel() {}

PyMODINIT_FUNC initmathmodule(void)
{
	(void) Py_InitModule("mathmodule", MathModuleMethods);
	import_array(); // for NumPy
	
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
		Each block has to evaluate its the dotproduct of
		blockDim.x elements by blockDim.x
		
		Once it is done it adds the value to dot_result
	*/
	__shared__ double cache[MAX_THREADS];
	
	int cacheIdx = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= d_vect1.getSize())
	{
		cache[cacheIdx] = 0;
		return;
	}
	
	cache[cacheIdx] = d_vect1[i] * d_vect2[i];
	
	__syncthreads();
	
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
		PyErr_SetString(PyExc_ValueError, "In mathmodule_dotproduct: array vect1 must be defined");
		return NULL;
	}
	if (vect2 == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_dotproduct: array vect2 must be defined");
		return NULL;
	}
	
	// Check that objects are 'double' type and vectors
	if (not_doublevector(vect1)) return NULL;
	if (not_doublevector(vect2)) return NULL;
	
	// Check dimensions
	const unsigned long int n = vect1->dimensions[0];
	if (vect2->dimensions[0] != n)
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_dotproduct: arrays vect1 and vect2 must have the same dimensions");
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
	dot_kernel<<<(n + MAX_THREADS -1)/MAX_THREADS, MAX_THREADS, MAX_THREADS * sizeof(double)>>>(d_vect1, d_vect2, d_dot_result);
	__stop()
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		cudaFree(d_dot_result);
		PyErr_SetString(PyExc_RuntimeError, "In mathmodule_dotproduct: CUDA failed");
		return NULL;
	}
	
	double dot_result;
	cudaMemcpy(&dot_result, d_dot_result, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_dot_result);
	
	// Return dot-product result
	return Py_BuildValue("d", dot_result);
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

int main(int argc, char *argv[])
{
	// Pass argv[0] to the Python interpreter
	Py_SetProgramName(argv[0]);
	
	// Initialize the Python interpreter. Required.
    	Py_Initialize();
	
	// Add a static module
	initmathmodule();
}


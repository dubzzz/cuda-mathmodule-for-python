#include <Python.h> // always first
#include "product.hpp"

#include "checks/CudaChecks.hpp"
#include "shared.hpp"

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

Vector *product(const Matrix &d_mat, const Vector &d_vect)
{
	if (d_mat.getWidth() != d_vect.getSize())
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_product: dim1 of mat must be equal to dim0 of vect");
		return NULL;
	}
	
	Vector *d_vect_result = new Vector(d_mat.getHeight());
	d_vect_result->memsetZero();
	
	const dim3 num_threads(1, MAX_THREADS, 1);
	const dim3 num_blocks((int)d_mat.getHeight(), ((int)d_mat.getWidth() + MAX_THREADS -1)/MAX_THREADS, 1);
	
	product_kernel<<<num_blocks, num_threads>>>(d_mat, d_vect, *d_vect_result);
	cudaThreadSynchronize(); // block until the device is finished
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		delete d_vect_result;
		PyErr_SetString(PyExc_RuntimeError, "In mathmodule_product: CUDA failed");
		return NULL;
	}
	
	return d_vect_result;
}


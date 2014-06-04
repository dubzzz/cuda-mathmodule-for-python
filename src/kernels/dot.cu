#include "dot.hpp"

#include "checks/CudaChecks.hpp"
#include "shared.hpp"

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

double dot(const Vector &d_vect1, const Vector &d_vect2)
{__CLOG__
	if (d_vect1.getSize() != d_vect2.getSize())
	{
		//TODO PyErr_SetString(PyExc_ValueError, "In mathmodule_dot: vectors vect1 and vect2 must have the same dimensions");
		return 0.;
	}
	
	double *d_dot_result;
	cudaErrorCheck(cudaMalloc(&d_dot_result, sizeof(double)));
	cudaErrorCheck(cudaMemset(d_dot_result, 0, sizeof(double)));
	__START__
	dot_kernel<<<(d_vect1.getSize() + MAX_THREADS -1)/MAX_THREADS, MAX_THREADS>>>(d_vect1, d_vect2, d_dot_result);
	cudaThreadSynchronize(); // block until the device is finished
	__STOP__
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		cudaFree(d_dot_result);
		//TODO PyErr_SetString(PyExc_RuntimeError, "In mathmodule_dot: CUDA failed");
		return 0.;
	}
	
	double h_dot_result;
	cudaErrorCheck(cudaMemcpy(&h_dot_result, d_dot_result, sizeof(double), cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaFree(d_dot_result));
	return h_dot_result;
}


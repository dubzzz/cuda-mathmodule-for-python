#include <Python.h>
#include "add.hpp"

#include "checks/CudaChecks.hpp"
#include "shared.hpp"

__global__ void add_kernel(const Vector d_va, const Vector d_vb, Vector d_vc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= d_va.getSize())
		return;
	
	d_vc[i] = d_va[i] + d_vb[i];
}

bool add(const Vector &d_va, const Vector &d_vb, Vector &d_vc)
{__CLOG__
	if (d_va.getSize() != d_vb.getSize() || d_va.getSize() != d_vc.getSize())
	{
		PyErr_SetString(PyExc_ValueError, "In mathmodule_add: vectors va, vb and vc must have the same dimensions");
		return false;
	}
	
	__START__
	add_kernel<<<(d_va.getSize() + MAX_THREADS -1)/MAX_THREADS, MAX_THREADS>>>(d_va, d_vb, d_vc);
	cudaThreadSynchronize(); // block until the device is finished
	__STOP__
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		PyErr_SetString(PyExc_RuntimeError, "In mathmodule_add: CUDA failed");
		return false;
	}
	
	return true;
}


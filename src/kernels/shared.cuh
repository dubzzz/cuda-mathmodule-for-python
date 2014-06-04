#ifndef __SHARED_CUH__
#define __SHARED_CUH__

#include "../preproc.hpp"
#define MAX_THREADS 256

#ifdef _DEBUG
	#define __CLOG__ __LOG__; cudaEvent_t start, stop;
	#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
	#define __STOP__ cudaEventRecord(stop, 0); cudaEventSynchronize(stop); float time; cudaEventElapsedTime(&time, start, stop); cudaEventDestroy(start); cudaEventDestroy(stop); printf("In %s at line %d\t%s\tTOOK: %fms\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, time);
#else
	#define __CLOG__
	#define __START__
	#define __STOP__
#endif

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

#endif

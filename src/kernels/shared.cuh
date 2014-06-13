#ifndef __SHARED_CUH__
#define __SHARED_CUH__

#include "../preproc.hpp"
#define MAX_THREADS 256

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

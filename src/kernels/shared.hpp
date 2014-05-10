#ifndef __SHARED_HPP__
#define __SHARED_HPP__

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

__device__ double atomicAdd(double* address, double val);

#endif


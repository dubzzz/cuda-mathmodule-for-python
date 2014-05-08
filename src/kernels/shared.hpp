#ifndef __SHARED_HPP__
#define __SHARED_HPP__

#define MAX_THREADS 1024

__device__ double atomicAdd(double* address, double val);

#endif

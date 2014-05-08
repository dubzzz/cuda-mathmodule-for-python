#include <iostream>
#include <stdio.h>

#include "CudaChecks.hpp"

void cudaAssert(const cudaError err, const char *file, const int line)
{ 
	if(cudaSuccess != err)
	{                                                
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
		exit(1);
	} 
}


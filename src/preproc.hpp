#ifdef _DEBUG
	#include <stdio.h>
	#include <typeinfo>
	#define __LOG__ {printf("In %s at line %d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);}
	#define __LOG(X) {printf("In %s at line %d\t%s:%ld\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, (long int)X);}
	#define __CLOG__ __LOG__; cudaEvent_t start, stop;
	#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
	#define __STOP__ cudaEventRecord(stop, 0); cudaEventSynchronize(stop); float time; cudaEventElapsedTime(&time, start, stop); cudaEventDestroy(start); cudaEventDestroy(stop); printf("In %s at line %d\t%s\tTOOK: %fms\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, time);
#else
	#define __LOG__   {}
	#define __LOG(X)  {}
	#define __CLOG__  {}
	#define __START__ {}
	#define __STOP__  {}
#endif


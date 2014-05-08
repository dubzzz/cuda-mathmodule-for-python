#ifndef __ERRORS_CUH__
#define __ERRORS_CUH__

#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }
void cudaAssert(const cudaError err, const char *file, const int line);

#endif


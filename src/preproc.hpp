#ifdef _DEBUG
	#include <stdio.h>
	#include <typeinfo>
	#define __LOG__ {printf("In %s at line %d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);}
	#define __LOG(X) {printf("In %s at line %d\t%s:%ld\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, (long int)X);}
#else
	#define __LOG__ {}
	#define __LOG(X) {}
#endif


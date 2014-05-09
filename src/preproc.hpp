#ifdef _DEBUG
	#include <stdio.h>
	#define __LOG__ {printf("In %s at line %d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);}
	#define __LOG(X) {printf("In %s at line %d\t%s:%d", __FILE__, __LINE__, __PRETTY_FUNCTION__, X);}
#else
	#define __LOG__ {}
	#define __LOG(X) {}
#endif


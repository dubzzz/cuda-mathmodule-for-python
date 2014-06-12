#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

#include <Python.h>
#include <numpy/arrayobject.h>

#ifndef __device__
	#define __device__
#endif

#ifndef __host__
	#define __host__
#endif
/*
	This class represents Vectors GPU-side
	It automatically frees memory when necessary
	
	/!\ No deep copy GPU to GPU is possible for the moment
*/

class Vector {
private:
	unsigned int size_;
	
	int *smart_ptr_counter_; // num of instances of Vector which share data_
	double *data_;

public:
	Vector(const unsigned int &size);
	Vector(const Vector &v);
	Vector(const double *h_v, const unsigned int &size);
	
	~Vector();
	void free();
	
	void memsetZero();
	
	PyArrayObject *toNumPy();
	__device__ double& get(const unsigned int &x) const;
	__device__ double& operator[](const unsigned int &x) const;
	__device__ __host__ unsigned int getSize() const;
    
    void __iadd__(Vector *vother);
    void __add__(Vector *v1, Vector *v2);
    
    double __dot__(Vector *vother);
};

/*
Call this method before calling anything else
if another method which requires NumPy is called without having called this method,
the program will return a segmentation fault
*/
void init_vector();

#endif


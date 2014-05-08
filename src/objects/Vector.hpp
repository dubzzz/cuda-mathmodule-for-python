#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

/*
	This class represents Vectors GPU-side
	It automatically frees memory when necessary
	
	/!\ No deep copy GPU to GPU is possible for the moment
*/

class Vector {
PyObject_HEAD
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
	__device__ __host__ double& get(const unsigned int &x) const;
	__device__ __host__ double& operator[](const unsigned int &x) const;
	__device__ __host__ unsigned int getSize() const;
};

#endif


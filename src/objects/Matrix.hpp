#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <Python.h>
#include <arrayobject.h>

#ifndef __device__
	#define __device__
#endif

#ifndef __host__
	#define __host__
#endif

class Matrix {
private:
	const unsigned int width_;
	const unsigned int height_;
	
	int *smart_ptr_counter_; // num of instances of Matrix which share data_
	double *data_;

public:
	
	Matrix(const unsigned int &height, const unsigned int &width);
	Matrix(const Matrix &m);
	Matrix(const double *h_m, const unsigned int &height, const unsigned int &width);
	
	~Matrix();
	void free();
	
	void memsetZero();
	
	__device__ double& get(const unsigned int &i, const unsigned int &j) const;
	__device__ __host__ unsigned int getWidth() const;
	__device__ __host__ unsigned int getHeight() const;
};

/*
	Call this method before calling anything else
	
	if another method which requires NumPy is called without having called this method,
	the program will return a segmentation fault
*/
void init_matrix();

#endif


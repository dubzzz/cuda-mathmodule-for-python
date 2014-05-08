#ifndef __VECTOR_CUH__
#define __VECTOR_CUH__

/*
	This class represents Vectors GPU-side
	It automatically frees memory when necessary
	
	/!\ No deep copy GPU to GPU is possible for the moment
	/!\ No copy to CPU is possible for the moment
*/

#include <string.h>
#include <iostream>

class Vector {
PyObject_HEAD
private:
	unsigned int size_;
	
	int *smart_ptr_counter_; // num of instances of Vector which share data_
	double *data_;

public:
	Vector(const unsigned int &size) : size_(size) {
		smart_ptr_counter_ = new int(1);
		cudaMalloc(&data_, size_ * sizeof(double));
	}
	
	Vector(const Vector &v) : size_(v.size_), data_(v.data_), smart_ptr_counter_(v.smart_ptr_counter_) {
		(*smart_ptr_counter_) += 1;
	}
	
	Vector(const double *h_v, const unsigned int &size) : size_(size) {
		smart_ptr_counter_ = new int(1);
		cudaMalloc(&data_, size_ * sizeof(double));
		cudaMemcpy(data_, h_v, size_ * sizeof(double), cudaMemcpyHostToDevice);
	}
	
	~Vector() {
		if (! data_) return;
		
		if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
			(*smart_ptr_counter_) -= 1;
			return;
		}
		
		delete smart_ptr_counter_;
		
		cudaFree(data_);
	}
	
	void free() {
		if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
			(*smart_ptr_counter_) -= 1;
			data_ = 0;
			return;
		}
		
		delete smart_ptr_counter_;
		
		cudaFree(data_);
		data_ = 0;
	}		
	
	void memsetZero() {
		cudaMemset(data_, 0, size_ * sizeof(double));
	}
	
	PyArrayObject *toNumPy() {
		int dims[] = {size_};
		PyArrayObject *h_arrayNumPy = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_DOUBLE);
		cudaErrorCheck(cudaMemcpy(h_arrayNumPy->data, data_, size_ * sizeof(double), cudaMemcpyDeviceToHost));
		return h_arrayNumPy;
	}
	
	__device__ __host__ double& get(const unsigned int &x) const {
		return data_[x];
	}
	
	__device__ __host__ double& operator[](const unsigned int &x) const {
		return data_[x];
	}
	
	__device__ __host__ unsigned int getSize() const { return size_; }
};

#endif


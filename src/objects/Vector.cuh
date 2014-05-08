#ifndef __VECTOR_CUH__
#define __VECTOR_CUH__

#include "Vector.hpp"

Vector::Vector(const unsigned int &size) : size_(size) {
	smart_ptr_counter_ = new int(1);
	cudaMalloc(&data_, size_ * sizeof(double));
}

Vector::Vector(const Vector &v) : size_(v.size_), data_(v.data_), smart_ptr_counter_(v.smart_ptr_counter_) {
	(*smart_ptr_counter_) += 1;
}

Vector::Vector(const double *h_v, const unsigned int &size) : size_(size) {
	smart_ptr_counter_ = new int(1);
	cudaMalloc(&data_, size_ * sizeof(double));
	cudaMemcpy(data_, h_v, size_ * sizeof(double), cudaMemcpyHostToDevice);
}

Vector::~Vector() {
	if (! data_) return;
	
	if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
		(*smart_ptr_counter_) -= 1;
		return;
	}
	
	delete smart_ptr_counter_;
	
	cudaFree(data_);
}

void Vector::free() {
	if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
		(*smart_ptr_counter_) -= 1;
		data_ = 0;
		return;
	}
	
	delete smart_ptr_counter_;
	
	cudaFree(data_);
	data_ = 0;
}		

void Vector::memsetZero() {
	cudaMemset(data_, 0, size_ * sizeof(double));
}

PyArrayObject *Vector::toNumPy() {
	int dims[] = {size_};
	PyArrayObject *h_arrayNumPy = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_DOUBLE);
	cudaErrorCheck(cudaMemcpy(h_arrayNumPy->data, data_, size_ * sizeof(double), cudaMemcpyDeviceToHost));
	return h_arrayNumPy;
}

__device__ __host__ double& Vector::get(const unsigned int &x) const {
	return data_[x];
}

__device__ __host__ double& Vector::operator[](const unsigned int &x) const {
	return data_[x];
}

__device__ __host__ unsigned int Vector::getSize() const { return size_; }

#endif


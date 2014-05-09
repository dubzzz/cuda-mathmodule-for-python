#include "../checks/CudaChecks.hpp"
#include "Vector.hpp"
#include <iostream>
#include "../preproc.hpp"

void init_vector() {
	__LOG__
	import_array();
}

Vector::Vector(const unsigned int &size) : size_(size) {
	__LOG__
	smart_ptr_counter_ = new int(1);
	cudaErrorCheck(cudaMalloc(&data_, size_ * sizeof(double)));
}

Vector::Vector(const Vector &v) : size_(v.size_), data_(v.data_), smart_ptr_counter_(v.smart_ptr_counter_) {
	__LOG__
	(*smart_ptr_counter_) += 1;
}

Vector::Vector(const double *h_v, const unsigned int &size) : size_(size) {
	__LOG__
	smart_ptr_counter_ = new int(1);
	cudaErrorCheck(cudaMalloc(&data_, size_ * sizeof(double)));
	cudaErrorCheck(cudaMemcpy(data_, h_v, size_ * sizeof(double), cudaMemcpyHostToDevice));
}

Vector::~Vector() {
	__LOG__
	if (! data_)
		return;
	
	if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
		(*smart_ptr_counter_) -= 1;
		return;
	}
	
	delete smart_ptr_counter_;
	cudaErrorCheck(cudaFree(data_));
}

void Vector::free() {
	__LOG__
	if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
		(*smart_ptr_counter_) -= 1;
		data_ = 0;
		return;
	}
	
	delete smart_ptr_counter_;
	
	cudaErrorCheck(cudaFree(data_));
	data_ = 0;
}		

void Vector::memsetZero() {
	__LOG__
	cudaErrorCheck(cudaMemset(data_, 0, size_ * sizeof(double)));
}

PyArrayObject *Vector::toNumPy() {
	__LOG__
	int dims[] = {size_};
	PyArrayObject *h_arrayNumPy = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_DOUBLE);
	cudaErrorCheck(cudaMemcpy(h_arrayNumPy->data, data_, size_ * sizeof(double), cudaMemcpyDeviceToHost));
	return h_arrayNumPy;
}

__device__ double& Vector::get(const unsigned int &x) const {
	return data_[x];
}

__device__ double& Vector::operator[](const unsigned int &x) const {
	return data_[x];
}

__device__ unsigned int Vector::getSize() const { return size_; }


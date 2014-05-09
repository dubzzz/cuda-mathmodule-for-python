#include "Matrix.hpp"
#include "../checks/CudaChecks.hpp"
#include "../preproc.hpp"

void init_matrix()
{__LOG__
	import_array();
}

Matrix::Matrix(const unsigned int &height, const unsigned int &width) : width_(width), height_(height)
{__LOG__
	smart_ptr_counter_ = new int(1);
	cudaErrorCheck(cudaMalloc((void **) &data_, width_ * height_ * sizeof(double)));
}

Matrix::Matrix(const Matrix &m) : width_(m.width_), height_(m.height_), data_(m.data_), smart_ptr_counter_(m.smart_ptr_counter_)
{__LOG__
	(*smart_ptr_counter_) += 1;
}

Matrix::Matrix(const double *h_m, const unsigned int &height, const unsigned int &width) : width_(width), height_(height)
{__LOG__
	smart_ptr_counter_ = new int(1);
	
	cudaErrorCheck(cudaMalloc((void **) &data_, width_ * height_ * sizeof(double)));
	cudaErrorCheck(cudaMemcpy(data_, h_m, width_ * height_ * sizeof(double), cudaMemcpyHostToDevice));
}

Matrix::~Matrix()
{__LOG__
	if (! data_) return;
	
	if (*smart_ptr_counter_ > 1) {
		(*smart_ptr_counter_) -= 1;
		return;
	}
	
	delete smart_ptr_counter_;
	
	cudaErrorCheck(cudaFree(data_));
}

void Matrix::free()
{__LOG__
	if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
		(*smart_ptr_counter_) -= 1;
		data_ = 0;
		return;
	}
	
	delete smart_ptr_counter_;
	
	cudaErrorCheck(cudaFree(data_));
	data_ = 0;
}

void Matrix::memsetZero()
{__LOG__
	cudaErrorCheck(cudaMemset(data_, 0, width_ * height_ * sizeof(double)));
}

__device__ double& Matrix::get(const unsigned int &i, const unsigned int &j) const {
	return data_[i * width_ + j];
}

__device__ __host__ unsigned int Matrix::getWidth() const { return width_; }
__device__ __host__ unsigned int Matrix::getHeight() const { return height_; }


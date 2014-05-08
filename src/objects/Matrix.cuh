#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

/*
http://stackoverflow.com/questions/11994010/in-cuda-how-can-we-call-a-device-function-in-another-translation-unit
	* CUDA 4.2 and does not support static linking so device functions must be defined in the same compilation unit. A common technique is to write the device function in a .cuh file and include it in the .cu file.
	* CUDA 5.0 supports a new feature called separate compilation. The CUDA 5.0 VS msbuild rules should be available in the CUDA 5.0 RC download.
*/

class Matrix {
private:
	const unsigned int width_;
	const unsigned int height_;
	
	int *smart_ptr_counter_; // num of instances of Matrix which share data_
	double *data_;

public:
	
	Matrix(const unsigned int &height, const unsigned int &width) : width_(width), height_(height) {
		smart_ptr_counter_ = new int(1);
		cudaErrorCheck(cudaMalloc((void **) &data_, width_ * height_ * sizeof(double)));
	}
	
	Matrix(const Matrix &m) : width_(m.width_), height_(m.height_), data_(m.data_), smart_ptr_counter_(m.smart_ptr_counter_) {
		(*smart_ptr_counter_) += 1;
	}
	
	Matrix(const double *h_m, const unsigned int &height, const unsigned int &width) : width_(width), height_(height) {
		smart_ptr_counter_ = new int(1);
		
		cudaErrorCheck(cudaMalloc((void **) &data_, width_ * height_ * sizeof(double)));
		cudaErrorCheck(cudaMemcpy(data_, h_m, width_ * height_ * sizeof(double), cudaMemcpyHostToDevice));
	}
	
	~Matrix() {
		if (! data_) return;
		
		if (*smart_ptr_counter_ > 1) {
			(*smart_ptr_counter_) -= 1;
			return;
		}
		
		delete smart_ptr_counter_;
		
		cudaErrorCheck(cudaFree(data_));
	}
	
	void free() {
		if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
			(*smart_ptr_counter_) -= 1;
			data_ = 0;
			return;
		}
		
		delete smart_ptr_counter_;
		
		cudaErrorCheck(cudaFree(data_));
		data_ = 0;
	}
	
	void memsetZero() {
		cudaErrorCheck(cudaMemset(data_, 0, width_ * height_ * sizeof(double)));
	}
	
	__device__ double& get(const unsigned int &i, const unsigned int &j) const {
		return data_[i * width_ + j];
	}
	
	__device__ __host__ unsigned int getWidth() const { return width_; }
	__device__ __host__ unsigned int getHeight() const { return height_; }
};

#endif


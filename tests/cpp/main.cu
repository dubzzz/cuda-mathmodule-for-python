#include <iostream>
#include <string>
#include "objects/Vector.hpp"

#define __START_CHRONO__ {cudaEventRecord(start, 0);}
#define __STOP_CHRONO(X) {cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(X, start, stop);}
#define NUM_TESTS 3

std::string measure_python(const std::string &popen_cmd)
{
    FILE* pipe = popen(popen_cmd.c_str(), "r");
    
    if (!pipe)
        return "ERROR";
    
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
        if(fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }
    pclose(pipe);
    return result;
}

void generate_random_vector(const unsigned int &size,
                            Vector *vout,
                            float *ptime_random_cpu,
                            float *ptime_cpu_to_gpu)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    __START_CHRONO__
    double data[size];
    for (unsigned int i(0) ; i != size ; i++)
        data[i] = rand();
    __STOP_CHRONO(ptime_random_cpu)
    
    __START_CHRONO__
    *vout = Vector(data, size);
    __STOP_CHRONO(ptime_cpu_to_gpu)
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv)
{
    srand((unsigned int) time(NULL));
    
    unsigned int tests_size[NUM_TESTS] = {1024, 8192, 65536};
    for (unsigned int i(0) ; i != NUM_TESTS ; i++)
    {
        std::cout << "\nBENCHMARK -- #tests= 1 - size= " << tests_size[i] << std::endl << std::endl;
        
        // Define useful variables
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        float time_init(0.);
        float time_kernel(0.);
        float time_random_cpu_1(0.), time_random_cpu_2(0.);
        float time_cpu_to_gpu_1(0.), time_cpu_to_gpu_2(0.);
        char popen_cmd[200];
        
        Vector v1(0), v2(0);
        
        // Vector::__add__
        
        std::cout << "Vector::__add__" << std::endl;
        
        generate_random_vector(tests_size[i], &v1, &time_random_cpu_1, &time_cpu_to_gpu_1);
        generate_random_vector(tests_size[i], &v2, &time_random_cpu_2, &time_cpu_to_gpu_2);
        
        __START_CHRONO__
        Vector vadd(tests_size[i]);
        __STOP_CHRONO(&time_init)
        
        __START_CHRONO__
        vadd.__add__(&v1, &v2);
        __STOP_CHRONO(&time_kernel)
        
        std::cout << "> Expected for CUDA: " << time_init+time_kernel << std::endl;
        std::cout << "\tRandom v1        [CPU]: " << time_random_cpu_1 << "ms" << std::endl;
        std::cout << "\tCopy v1     [CPU->GPU]: " << time_cpu_to_gpu_1 << "ms" << std::endl;
        std::cout << "\tRandom v2        [CPU]: " << time_random_cpu_2 << "ms" << std::endl;
        std::cout << "\tCopy v2     [CPU->GPU]: " << time_cpu_to_gpu_2 << "ms" << std::endl;
        std::cout << "\tCreate vadd      [GPU]: " << time_init << "ms" << std::endl;
        std::cout << "\tVector::__add__  [GPU]: " << time_kernel << "ms" << std::endl;
        
        sprintf(popen_cmd, "python test.py add %d cuda", tests_size[i]);
        std::cout << "> CUDA             : " << measure_python(popen_cmd);
        
        sprintf(popen_cmd, "python test.py add %d numpy", tests_size[i]);
        std::cout << "> NumPy            : " << measure_python(popen_cmd) << std::endl;
        
        // Vector::__iadd__
        
        std::cout << "Vector::__iadd__" << std::endl;
        
        generate_random_vector(tests_size[i], &v1, &time_random_cpu_1, &time_cpu_to_gpu_1);
        generate_random_vector(tests_size[i], &v2, &time_random_cpu_2, &time_cpu_to_gpu_2);
        
        __START_CHRONO__
        v1.__iadd__(&v2);
        __STOP_CHRONO(&time_kernel)
        
        std::cout << "> Expected for CUDA: " << time_kernel << std::endl;
        std::cout << "\tRandom v1        [CPU]: " << time_random_cpu_1 << "ms" << std::endl;
        std::cout << "\tCopy v1     [CPU->GPU]: " << time_cpu_to_gpu_1 << "ms" << std::endl;
        std::cout << "\tRandom v2        [CPU]: " << time_random_cpu_2 << "ms" << std::endl;
        std::cout << "\tCopy v2     [CPU->GPU]: " << time_cpu_to_gpu_2 << "ms" << std::endl;
        std::cout << "\tVector::__iadd__ [GPU]: " << time_kernel << "ms" << std::endl;
        
        sprintf(popen_cmd, "python test.py iadd %d cuda", tests_size[i]);
        std::cout << "> CUDA             : " << measure_python(popen_cmd);
        sprintf(popen_cmd, "python test.py iadd %d numpy", tests_size[i]);
        std::cout << "> NumPy            : " << measure_python(popen_cmd) << std::endl;
        
        // Free memory
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}


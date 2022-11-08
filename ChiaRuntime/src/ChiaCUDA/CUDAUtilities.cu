#include "ChiaCUDA/CUDAUtilities.cuh"

#include <iostream>

bool CheckCUDAStatus(const cudaError_t &error, const char *fileName, unsigned int line)
{
    return error == cudaSuccess;
}

void PrintCUDAErrorMessage(const cudaError_t &error, bool newline)
{
    if (error == cudaSuccess)
        return;
    std::printf("%s", cudaGetErrorString(error));
    if (newline)
        std::printf("\n");
}

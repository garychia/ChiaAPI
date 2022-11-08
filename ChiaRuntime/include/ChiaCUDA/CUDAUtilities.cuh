#ifndef CUDAUTILITIES_CUH
#define CUDAUTILITIES_CUH

#include <cuda_runtime.h>

namespace ChiaRuntime
{
namespace ChiaCUDA
{
/**
 * @brief Check if a CUDA function executed successfully
 *
 * @param error the value returned by the CUDA function.
 * @return true if the CUDA function executed successfully.
 * @return false otherwise.
 */
bool CheckCUDASuccessful(const cudaError_t &error);

/**
 * @brief Print the error message based on the error code.
 *
 * @param error the result returned by a CUDA function.
 * @param newline a bool indicates whether a new-line character will be printed after the message.
 */
void PrintCUDAErrorMessage(const cudaError_t &error, bool newline = false);

} // namespace ChiaCUDA
} // namespace ChiaRuntime

#endif

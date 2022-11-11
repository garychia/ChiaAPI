#ifndef ARITHMETIC_CUH
#define ARITHMETIC_CUH

#include <cuda_runtime.h>

#include "CUDAUtilities.cuh"

namespace ChiaAlgs
{
namespace ChiaCUDA
{
/**
 * @brief Perform element-wise addition on two arrays
 *
 * @tparam OutputType the element type of the array that will store the result.
 * @tparam FirstOperandType the element type of the array that is the first
 * operand.
 * @tparam SecondOperandType the element type of the array that is the second
 * operand.
 * @tparam IndexType the type of the index.
 * @param dest the array used to store the result.
 * @param operand1 the array that represents the first operand.
 * @param operand2 the array that represents the second operand.
 * @param size the size of the arrays.
 */
template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>
__global__ void ArrayAddition(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2,
                              const IndexType size);

/**
 * @brief Perform element-wise subtraction on two arrays
 *
 * @tparam OutputType the element type of the array that will store the result.
 * @tparam FirstOperandType the element type of the array that is the first
 * operand.
 * @tparam SecondOperandType the element type of the array that is the second
 * operand.
 * @tparam IndexType the type of the index.
 * @param dest the array used to store the result.
 * @param operand1 the array that represents the first operand.
 * @param operand2 the array that represents the second operand.
 * @param size the size of the arrays.
 */
template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>
__global__ void ArraySubtraction(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2,
                                 const IndexType size);

/**
 * @brief Perform element-wise multiplication on two arrays
 *
 * @tparam OutputType the element type of the array that will store the result.
 * @tparam FirstOperandType the element type of the array that is the first
 * operand.
 * @tparam SecondOperandType the element type of the array that is the second
 * operand.
 * @tparam IndexType the type of the index.
 * @param dest the array used to store the result.
 * @param operand1 the array that represents the first operand.
 * @param operand2 the array that represents the second operand.
 * @param size the size of the arrays.
 */
template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>
__global__ void ArrayMultiplication(OutputType *dest, const FirstOperandType *operand1,
                                    const SecondOperandType *operand2, const IndexType size);

/**
 * @brief Perform element-wise division on two arrays
 *
 * @tparam OutputType the element type of the array that will store the result.
 * @tparam FirstOperandType the element type of the array that is the first
 * operand.
 * @tparam SecondOperandType the element type of the array that is the second
 * operand.
 * @tparam IndexType the type of the index.
 * @param dest the array used to store the result.
 * @param operand1 the array that represents the first operand.
 * @param operand2 the array that represents the second operand.
 * @param size the size of the arrays.
 */
template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>
__global__ void ArrayDivision(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2,
                              const IndexType size);

template <class OutputType, class ArrayType, class ScalerType, class IndexType>
__global__ void ArrayScalerAddition(OutputType *dest, const ArrayType *arr, const ScalerType scaler,
                                    const IndexType size);

template <class OutputType, class ArrayType, class ScalerType, class IndexType>
__global__ void ArrayScalerSubtraction(OutputType *dest, const ArrayType *arr, const ScalerType scaler,
                                       const IndexType size);

template <class OutputType, class ArrayType, class ScalerType, class IndexType>
__global__ void ArrayScalerMultiplication(OutputType *dest, const ArrayType *arr, const ScalerType scaler,
                                          const IndexType size);

template <class OutputType, class ArrayType, class ScalerType, class IndexType>
__global__ void ArrayScalerDivision(OutputType *dest, const ArrayType *arr, const ScalerType scaler,
                                    const IndexType size);

/**
 * @brief Map each element in a device array to a new value
 * 
 * @tparam T the type of elements in the output array.
 * @tparam U the type of elements in the input array.
 * @tparam MapFunction the type of function that maps each element in the input array to a new value.
 * @param output an array where the mapped elements are stored.
 * @param input an array where each element is mapped by f.
 * @param f a mapping function.
 * @param size the number of elements in the input array.
 */
template <class T, class U, class MapFunction>
__global__ void ArrayMap(T *output, const U *input, MapFunction &&f, size_t size);

/**
 * @brief Calculate each element of an array raised to a given power
 *
 * @tparam T the type of the array element.
 * @tparam IndexType the type of the index.
 * @tparam PowerType the type of the power.
 * @param dest the destination array where the results will be stored.
 * @param arr an array.
 * @param size the number of elements in the array.
 * @param power the power.
 */
template <class T, class IndexType, class PowerType>
__global__ void Power(T *dest, const T *arr, IndexType size, PowerType power);

/**
 * @brief Calculate the summation of all the elements in an array on the device
 * 
 * @tparam T the type of the elements in the array.
 * @tparam IndexType the type of the index.
 * @param arr a device array.
 * @param size the number of elements in the array.
 * @return T the summation of the elements.
 */
template <class T, class IndexType> T SumDeviceArray(const T *arr, IndexType size);

} // namespace ChiaCUDA
} // namespace ChiaAlgs

#include "Math.hpp"

namespace ChiaAlgs
{
namespace ChiaCUDA
{
#define ARRAY_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, op)                                                        \
    template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>                      \
    __global__ void func_name(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2,   \
                              const IndexType size)                                                                    \
    {                                                                                                                  \
        const size_t i = threadIdx.x + blockIdx.x * blockDim.x;                                                        \
        if (i < size)                                                                                                  \
            dest[i] = operand1[i] op operand2[i];                                                                      \
    }

#define ARRAY_SCALER_FUNCTION_IMPLEMENTATION(func_name, op)                                                            \
    template <class OutputType, class ArrayType, class ScalerType, class IndexType>                                    \
    __global__ void func_name(OutputType *dest, const ArrayType *arr, const ScalerType scaler, const IndexType size)   \
    {                                                                                                                  \
        const size_t i = threadIdx.x + blockIdx.x * blockDim.x;                                                        \
        if (i < size)                                                                                                  \
            dest[i] = arr[i] op scaler;                                                                                \
    }

ARRAY_ARITHMETIC_FUNCTION_IMPLEMENTATION(ArrayAddition, +);
ARRAY_ARITHMETIC_FUNCTION_IMPLEMENTATION(ArraySubtraction, -);
ARRAY_ARITHMETIC_FUNCTION_IMPLEMENTATION(ArrayMultiplication, *);
ARRAY_ARITHMETIC_FUNCTION_IMPLEMENTATION(ArrayDivision, /);

ARRAY_SCALER_FUNCTION_IMPLEMENTATION(ArrayScalerAddition, +);
ARRAY_SCALER_FUNCTION_IMPLEMENTATION(ArrayScalerSubtraction, -);
ARRAY_SCALER_FUNCTION_IMPLEMENTATION(ArrayScalerMultiplication, *);
ARRAY_SCALER_FUNCTION_IMPLEMENTATION(ArrayScalerDivision, /);

template <class T, class U, class MapFunction>
__global__ void ArrayMap(T *output, const U *input, MapFunction &&f, size_t size)
{
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        output[i] = f(input[i]);
}

template <class T, class IndexType, class PowerType>
__global__ void Power(T *dest, const T *arr, IndexType size, PowerType power)
{
    const IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        dest[i] = ChiaMath::Power<T, PowerType>(arr[i], power);
}

namespace
{
template <class T, class IndexType> __global__ void SumSubArray(T *dest, const T *arr, IndexType size)
{
    const IndexType start = threadIdx.x;
    const IndexType end = ChiaMath::Min(size, start + blockDim.x);
    for (IndexType i = start; i < end; i++)
        *dest = arr[i];
}
} // namespace

template <class T, class IndexType> T SumDeviceArray(const T *arr, IndexType size)
{
    if (0 == size)
        return 0;
    const size_t nThreads = 32;
    if (size <= nThreads)
    {
        T *pSum;
        ChiaRuntime::ChiaCUDA::PrintCUDAErrorMessage(cudaMalloc(&pSum, sizeof(T)));
        SumSubArray<<<1, nThreads>>>(pSum, arr, size);
        T result;
        ChiaRuntime::ChiaCUDA::PrintCUDAErrorMessage(cudaMemcpy(&result, pSum, sizeof(T), cudaMemcpyDeviceToHost));
        ChiaRuntime::ChiaCUDA::PrintCUDAErrorMessage(cudaFree(pSum));
        return result;
    }
    const auto halfSize = size >> 1;
    return SumDeviceArray(arr, halfSize) + SumDeviceArray(arr + halfSize, size - halfSize);
}
} // namespace ChiaCUDA
} // namespace ChiaAlgs

#endif // ARITHMETIC_CUH

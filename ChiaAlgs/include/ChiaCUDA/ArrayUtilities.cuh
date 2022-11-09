#ifndef ARRAY_UTILITIES_H
#define ARRAY_UTILITIES_H

namespace ChiaAlgs
{
namespace ChiaCUDA
{
template <class ArrayType, class SizeType, class ValueType>
__global__ void PopulateArray(ArrayType *arr, SizeType size, ValueType value)
{
    const SizeType i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        arr[i] = value;
}
} // namespace ChiaCUDA
} // namespace ChiaAlgs

#endif // ARRAY_UTILITIES_H

#ifndef CUDAARRAY_HPP
#define CUDAARRAY_HPP

#include "ChiaCUDA/ArrayUtilities.cuh"
#include "ChiaCUDA/CUDAUtilities.cuh"

#include <cuda_runtime.h>

using namespace ChiaRuntime::ChiaCUDA;

namespace ChiaData
{
namespace ChiaCUDA
{
/**
 * @brief CUDAArray represents an array on the CUDA device side
 *
 * @tparam T the type of the elements in the CUDAArray.
 */
template <class T> class CUDAArray
{
  private:
    // Data stored on the GPU side.
    T *deviceArr = nullptr;
    // Number of elements stored.
    size_t size = 0;

  public:
    /**
     * @brief Construct a new CUDAArray object
     */
    CUDAArray() : deviceArr(nullptr), size(0)
    {
    }

    /**
     * @brief Construct a new CUDAArray object
     *
     * @param size the initial size of the CUDAArray to be generated.
     * @param value the value the CUDAArray will be filled with.
     **/
    CUDAArray(size_t size, const T &value) : deviceArr(nullptr), size(size)
    {
        if (!size)
            return;
        PrintCUDAErrorMessage(cudaMalloc(&deviceArr, size * sizeof(T)));
        const size_t nThreads = 32;
        const size_t nBlocks = (size + nThreads - 1) / nThreads;
        ChiaAlgs::ChiaCUDA::PopulateArray<<<nBlocks, nThreads>>>(deviceArr, size, value);
    }

    /**
     * @brief Construct a new CUDAArray object
     *
     * @param size the size of the CUDAArray.
     * @param arr the host array whose elements will be copied to this CUDAArray.
     */
    CUDAArray(size_t size, const T *arr) : deviceArr(nullptr), size(size)
    {
        if (!size)
            return;
        PrintCUDAErrorMessage(cudaMalloc(&deviceArr, size * sizeof(T)));
        PrintCUDAErrorMessage(cudaMemcpy(deviceArr, arr, sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    /**
     * @brief Construct a new CUDAArray object
     *
     * @param other a CUDAArray to be copied.
     */
    CUDAArray(const CUDAArray<T> &other) : deviceArr(nullptr), size(other.size)
    {
        if (!size)
            return;
        PrintCUDAErrorMessage(cudaMalloc(&deviceArr, size * sizeof(T)));
        PrintCUDAErrorMessage(cudaMemcpy(deviceArr, other.deviceArr, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    }

    /**
     * @brief Construct a new CUDAArray object
     *
     * @param other a CUDAArray to be 'moved' into this CUDAArray.
     */
    CUDAArray(CUDAArray<T> &&other) noexcept : deviceArr(other.deviceArr), size(other.size)
    {
        other.size = 0;
        other.deviceArr = nullptr;
    }

    /**
     * @brief Destroy the CUDAArray object
     */
    virtual ~CUDAArray()
    {
        if (deviceArr)
            PrintCUDAErrorMessage(cudaFree(deviceArr));
    }

    /**
     * @brief CUDAArray Copy Assignment
     *
     * @param other a CUDAArray to be copied to this CUDArray.
     * @return CUDAArray<T>& this CUDAArray.
     */
    virtual CUDAArray<T> &operator=(const CUDAArray<T> &other)
    {
        if (deviceArr)
            PrintCUDAErrorMessage(cudaFree(deviceArr));
        size = other.size;
        PrintCUDAErrorMessage(cudaMalloc(&deviceArr, sizeof(T) * size));
        PrintCUDAErrorMessage(cudaMemcpy(deviceArr, other.deviceArr, sizeof(T) * size, cudaMemcpyDeviceToDevice));
        return *this;
    }

    /**
     * @brief CUDAArray Move Assignment
     *
     * @param other a CUDAArray to be 'moved' into this CUDAArray.
     * @return CUDAArray<T>& this CUDArray.
     */
    virtual CUDAArray<T> &operator=(CUDAArray<T> &&other)
    {
        if (deviceArr)
            PrintCUDAErrorMessage(cudaFree(deviceArr));
        deviceArr = nullptr;
        size = other.size;
        deviceArr = other.deviceArr;
        other.size = 0;
        other.deviceArr = nullptr;
        return *this;
    }

    /**
     * @brief Return the size (number of elements) of the CUDAArray
     *
     * @return size_t the number of elements this CUDAArray stores.
     */
    size_t Size() const
    {
        return size;
    }

    /**
     * @brief Check if this CUDAArray is empty or not.
     *
     * @return true if it is empty.
     * @return false if it has any element.
     */
    virtual bool IsEmpty() const
    {
        return !size;
    }

    /**
     * @brief Copy the data of the CUDAArray from the device to the host
     *
     * @param dest the destination array on the host side to which the data will be copied.
     */
    void CopyToHost(T *dest) const
    {
        PrintCUDAErrorMessage(cudaMemcpy(dest, deviceArr, sizeof(T) * size, cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Copy the data of the CUDAArray from the device to the host
     *
     * @param dest the destination array on the host side to which the data will be copied.
     * @param nElements the number of elements to be copied.
     */
    void CopyToHost(T *dest, size_t nElements) const
    {
        PrintCUDAErrorMessage(
            cudaMemcpy(dest, deviceArr, sizeof(T) * (nElements > size ? size : nElements), cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Copy the data of the CUDAArray from the host to the device
     *
     * @param src the source where data will be copied from the host to the device.
     */
    void CopyFromHost(const T *src)
    {
        PrintCUDAErrorMessage(cudaMemcpy(deviceArr, src, sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    /**
     * @brief Copy the data of the CUDAArray from the host to the device
     *
     * @param src the source where data will be copied from the host to the device.
     * @param nElements the number of elements to be copied.
     */
    void CopyFromHost(const T *src, size_t nElements)
    {
        PrintCUDAErrorMessage(
            cudaMemcpy(deviceArr, src, sizeof(T) * (nElements > size ? size : nElements), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Retrieve the pointer that points to the data of CUDAArray on the device side.
     *
     * @return T* a pointer to the data on the device side.
     */
    T *GetDevicePtr() const
    {
        return deviceArr;
    }
};
} // namespace ChiaCUDA
} // namespace ChiaData

#endif // CUDAARRAY_HPP

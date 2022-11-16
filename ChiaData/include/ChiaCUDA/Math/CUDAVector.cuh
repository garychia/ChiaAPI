#ifndef CUDAVECTOR_HPP
#define CUDAVECTOR_HPP

#include "ChiaCUDA/Arithmetic.cuh"
#include "ChiaCUDA/CUDAArray.cuh"
#include "Vector.hpp"

namespace ChiaData
{
namespace ChiaCUDA
{
namespace Math
{
template <class T> class CUDAVector
{
  private:
    // Buffer on the device
    CUDAArray<T> deviceBuffer;
    // Buffer on the host
    ChiaData::Math::Vector<T> hostBuffer;
    // Indicates whether the host buffer has the latest data from the device
    bool hostBufferUpdated;

    /**
     * @brief Clear the host buffer.
     */
    void ClearHostBuffer()
    {
        hostBuffer = Vector<T>();
        hostBufferUpdated = false;
    }

    /**
     * @brief Copy the data from the device to the host
     */
    void CopyFromDeviceToHost()
    {
        ClearHostBuffer();
        hostBufferUpdated = true;
        if (deviceBuffer.IsEmpty())
            return;
        hostBuffer = Vector<T>(deviceBuffer.Size());
        cudaArray.CopyToHost(&hostBuffer[0]);
    }

  public:
    /**
     * @brief Construct a new CUDAVector object
     */
    CUDAVector() : deviceBuffer(), hostBuffer(), hostBufferUpdated(true)
    {
    }

    /**
     * @brief Construct a new CUDAVector object
     *
     * @param size the number of elements to be stored in this CUDAVector.
     * @param value the value used to populate this CUDAVector.
     */
    CUDAVector(size_t size, const T &value = 0)
        : deviceBuffer(size, value), hostBuffer(size, value), hostBufferUpdated(true)
    {
    }

    /**
     * @brief Construct a new CUDAVector object
     *
     * @param lst an initializer_list whose elements will be copied to this CUDAVector.
     */
    CUDAVector(const std::initializer_list<T> &lst) : deviceBuffer(lst.size()), hostBuffer(lst), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(&hostBuffer[0]);
    }

    /**
     * @brief Construct a new CUDAVector object
     *
     * @tparam N the size of the array.
     * @param arr an array whose elements will be copied to this CUDAVector.
     */
    template <size_t N>
    CUDAVector(const std::array<T, N> &arr) : deviceBuffer(arr.size()), hostBuffer(arr), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(arr.data());
    }

    /**
     * @brief Construct a new CUDAVector object
     *
     * @param v a vector whose elements will be copied to this CUDAVector.
     */
    CUDAVector(const std::vector<T> &v) : deviceBuffer(v.size()), hostBuffer(v), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(v.data());
    }

    /**
     * @brief Construct a new CUDAVector object
     *
     * @param v a vector whose elements will be copied to this CUDAVector.
     */
    CUDAVector(const Vector<T> &v) : deviceBuffer(v.Size()), hostBuffer(v), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(hostBuffer.AsRawPointer());
    }

    /**
     * @brief Construct a new CUDAVector object
     *
     * @param v a vector that will be 'moved' into this CUDAVector.
     */
    CUDAVector(Vector<T> &&v) : deviceBuffer(v.Size()), hostBuffer(std::move(v)), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(hostBuffer.AsRawPointer());
    }

    /**
     * @brief Construct a new CUDAVector object
     *
     * @param v a CUDAVector that will be copied into this CUDAVector.
     */
    CUDAVector(const CUDAVector<T> &v) : deviceBuffer(v), hostBuffer(), hostBufferUpdated(false)
    {
    }

    /**
     * @brief Construct a new CUDAVector object
     *
     * @param v a CUDAVector to be 'moved' into this CUDAVector.
     */
    CUDAVector(CUDAVector<T> &&v) : deviceBuffer(std::move(v)), hostBuffer(), hostBufferUpdated(v.hostBufferUpdated)
    {
        if (v.hostBufferUpdated)
            hostBuffer = std::move(v.hostBuffer);
    }

    /**
     * @brief Destroy the CUDAVector object
     */
    virtual ~CUDAVector()
    {
        ClearHostBuffer();
    }

    /**
     * @brief CUDAVector Copy Assignment
     *
     * @param other a CUDAVector to be copied to this CUDAVector.
     * @return Vector<T>& this CUDAVector.
     */
    virtual CUDAVector<T> &operator=(const CUDAVector<T> &other)
    {
        if (&other == this)
            return *this;
        ClearHostBuffer();
        deviceBuffer = other.deviceBuffer;
        return *this;
    }

    /**
     * @brief CUDAVector Move Assignment
     *
     * @param other a CUDAVector to be 'moved' into this CUDAVector.
     * @return Vector<T>& this CUDAVector.
     */
    virtual Vector<T> &operator=(CUDAVector<T> &&other)
    {
        ClearHostBuffer();
        deviceBuffer = Types::Move(other.deviceBuffer);
        if (other.hostBufferUpdated)
        {
            hostBuffer = Types::Move(other.hostBuffer);
            hostBufferUpdated = true;
        }
        return *this;
    }

    /**
     * @brief Get the size (number of elements) of the CUDAVector
     *
     * @return size_t the size of the CUDAVector.
     */
    size_t Size() const
    {
        return deviceBuffer.Size();
    }

    /**
     * @brief Check if the CUDAVector is empty
     *
     * @return true if empty.
     * @return false if having any element.
     */
    bool IsEmpty() const
    {
        return deviceBuffer.IsEmpty();
    }

    /**
     * @brief Get the element at the given index in the CUDAVector
     *
     * @param index an index that is expected to be less than the size.
     * @return T& the element at the index.
     */
    T &operator[](size_t index)
    {
        if (!hostBufferUpdated)
            CopyFromDeviceToHost();
        return hostBuffer[index];
    }

    /**
     * @brief Get the element at the given index in the CUDAVector
     *
     * @param index an index that is expected to be less than the size.
     * @return const T& the element at the index.
     */
    const T &operator[](size_t index) const
    {
        return hostBuffer[index];
    }

    /**
     * @brief Return the dimension (number of elements) of the CUDAVector
     *
     * @return size_t the dimension.
     */
    size_t Dimension() const
    {
        return Size();
    }

    /**
     * @brief Calculate the summation of all the elements in this CUDAVector
     *
     * @return T the summation
     */
    T Sum() const
    {
        return ChiaAlgs::ChiaCUDA::SumArray(deviceBuffer.GetDevicePtr(), deviceBuffer.Size());
    }

    /**
     * @brief Calculate the Euclidean norm of the CUDAVector.
     *
     * @tparam ReturnType the type of the output.
     * @return ReturnType the Euclidean norm of this CUDAVector.
     */
    template <class ReturnType> ReturnType Length() const
    {
        return LpNorm<ReturnType>(2);
    }

    /**
     * @brief Return the Euclidean norm of the CUDAVector (Same as Vector::Length)
     *
     * @tparam ReturnType the type of the output.
     * @return ReturnType the Euclidean norm of this CUDAVector.
     */
    template <class ReturnType> ReturnType EuclideanNorm() const
    {
        return Length<ReturnType>();
    }

    template <class ReturnType> ReturnType LpNorm(size_t p) const
    {
        T *squaredArray;
        ChiaRuntime::ChiaCUDA::PrintCUDAErrorMessage(cudaMalloc(&squaredArray, sizeof(T) * deviceBuffer.Size()));
        const size_t nThreads = 32;
        const size_t nBlocks = (deviceBuffer.Size() + nThreads - 1) / nThreads;
        ChiaAlgs::ChiaCUDA::MapArray<<<nBlocks, nThreads>>>(
            squaredArray, deviceBuffer.GetDevicePtr(), [=](const T e) __device__ { return ChiaMath::Power(e, p); },
            deviceBuffer.Size());
        ChiaRuntime::ChiaCUDA::PrintCUDAErrorMessage(cudaDeviceSynchronize());
        T result = ChiaAlgs::ChiaCUDA::SumArray(squaredArray, deviceBuffer.Size());
        ChiaRuntime::ChiaCUDA::PrintCUDAErrorMessage(cudaFree(squaredArray));
        return ChiaMath::Power(result, 1 / p);
    }

    template <class ScalerType> auto Add(const ScalerType &scaler) const
    {
        CUDAVector<(*this)[0] + scaler> result(*this);
        const size_t nThreads = 32;
        const size_t nBlocks = (thisDimension + nThreads - 1) / nThreads;
        result.hostBufferUpdated = false;
        ChiaAlgs::ChiaCUDA::AddArrayScaler<<<nBlocks, nThreads>>>(result.deviceBuffer.GetDevicePtr(),
                                                                  deviceBuffer.GetDevicePtr(), scaler, Dimension());
        ChiaRuntime::ChiaCUDA::PrintCUDAErrorMessage(cudaDeviceSynchronize());
        return result;
    }

    template <class U> auto Add(const CUDAVector<U> &other) const
    {
        CUDAVector<(*this)[0] + other[0]> result(*this);
        if (IsEmpty() || other.IsEmpty())
            return result;
        const auto thisDimension = Dimension();
        const auto otherDimension = other.Dimension();
        else if (thisDimension % otherDimension != 0)
        {
            StringStream stream;
            stream << "CUDAVector - Invalid Argument:\n";
            stream << "CUDAVector::Add: the argument is expected to have a dimension that is a factor of that of this "
                      "CUDAVector.\n";
            stream << "Dimension of this CUDAVector: " << thisDimension << "\n";
            stream << "Dimension of the argument: " << otherDimension << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
        result.hostBufferUpdated = false;
        const size_t nThreads = 32;
        const size_t nBlocks = (thisDimension + nThreads - 1) / nThreads;
        ChiaAlgs::ChiaCUDA::AddArrays<<<nBlocks, nThreads>>>(result.deviceBuffer.GetDevicePtr(),
                                                             deviceBuffer.GetDevicePtr(),
                                                             other.deviceBuffer.GetDevicePtr(), otherDimension);
        ChiaRuntime::ChiaCUDA::PrintCUDAErrorMessage(cudaDeviceSynchronize());
        return result;
    }
};
} // namespace Math
} // namespace ChiaCUDA
} // namespace ChiaData

#endif

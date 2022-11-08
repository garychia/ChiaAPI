#ifndef CUDAVECTOR_HPP
#define CUDAVECTOR_HPP

#include "ChiaCUDA/CUDAArray.hpp"
#include "Vector.hpp"

namespace ChiaData
{
namespace ChiaCUDA
{
template <class T> class CUDAVector
{
  private:
    CUDAArray<T> deviceBuffer;

    Vector<T> hostBuffer;

    bool hostBufferUpdated;

    void CleanUp()
    {
        hostBuffer = Vector<T>();
        hostBufferUpdated = false;
    }

    void copyDataToHost()
    {
        CleanUp();
        hostBufferUpdated = true;
        if (deviceBuffer.IsEmpty())
            return;
        hostBuffer = Vector<T>(deviceBuffer.Size());
        cudaArray.CopyToHost(&hostBuffer[0]);
    }

  public:
    CUDAVector() : deviceBuffer(), hostBuffer(), hostBufferUpdated(true)
    {
    }

    CUDAVector(size_t size, const T &value = 0)
        : deviceBuffer(size, value), hostBuffer(size, value), hostBufferUpdated(true)
    {
    }

    CUDAVector(const std::initializer_list<T> &lst) : deviceBuffer(lst.size()), hostBuffer(lst), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(&hostBuffer[0]);
    }

    template <size_t N>
    CUDAVector(const std::array<T, N> &arr) : deviceBuffer(arr.size()), hostBuffer(arr), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(arr.data());
    }

    CUDAVector(const std::vector<T> &v) : deviceBuffer(v.size()), hostBuffer(v), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(v.data());
    }

    CUDAVector(const Vector<T> &v) : deviceBuffer(v.Size()), hostBuffer(v), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(hostBuffer.AsRawPointer());
    }

    CUDAVector(const Vector<T> &v) : deviceBuffer(v.Size()), hostBuffer(v), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(hostBuffer.AsRawPointer());
    }

    CUDAVector(Vector<T> &&v) : deviceBuffer(v.Size()), hostBuffer(std::move(v)), hostBufferUpdated(true)
    {
        deviceBuffer.CopyFromHost(hostBuffer.AsRawPointer());
    }

    CUDAVector(const CUDAVector<T> &v) : deviceBuffer(v), hostBuffer(), hostBufferUpdated(false)
    {
    }

    CUDAVector(CUDAVector<T> &&v) : deviceBuffer(std::move(v)), hostBuffer(), hostBufferUpdated(v.hostBufferUpdated)
    {
        if (v.hostBufferUpdated)
            hostBuffer = std::move(v.hostBuffer);
    }

    virtual ~CUDAVector()
    {
        CleanUp();
    }

    virtual Vector<T> &operator=(const CUDAVector<T> &other)
    {
        CleanUp();
        deviceBuffer = other.deviceBuffer;
        return *this;
    }

    virtual Vector<T> &operator=(CUDAVector<T> &&other)
    {
        CleanUp();
        deviceBuffer = std::move(other.deviceBuffer);
        if (other.hostBufferUpdated)
        {
            hostBuffer = std::move(other.hostBuffer);
            hostBufferUpdated = true;
        }
        return *this;
    }

    /**
     * @brief Get the size of the CUDAVector.
     *
     * @return size_t the size of the CUDAVector.
     */
    size_t Size() const
    {
        return deviceBuffer.Size();
    }

    /**
     * @brief Check if the CUDAVector is empty.
     *
     * @return true if empty.
     * @return false if having any element.
     */
    bool IsEmpty() const
    {
        return deviceBuffer.IsEmpty();
    }

    T &operator[](size_t index)
    {
        if (!hostBufferUpdated)
            copyDataToHost();
        return hostBuffer[index];
    }

    const T &operator[](size_t index) const
    {
        return hostBuffer[index];
    }

    size_t Dimension() const
    {
        return Size();
    }

    template <class ReturnType> ReturnType Length() const
    {
        if (Size() == 0)
            throw Exceptions::EmptyVector("Vector: Length of an empty vector is undefined.");
        return LpNorm<ReturnType>(2);
    }

    template <class ReturnType> ReturnType EuclideanNorm() const
    {
        if (Size() == 0)
            throw Exceptions::EmptyVector("Vector: Euclidean norm of an empty vector is undefined.");
        return Length<ReturnType>();
    }
};
} // namespace ChiaCUDA
} // namespace ChiaData

#endif

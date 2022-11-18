#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <initializer_list>

namespace ChiaData
{
/**
 * @brief Represents an array of a fix length.
 *
 * @tparam T the type of elements in the array.
 */
template <class T> class Array
{
  protected:
    /**
     * @brief where the elements are stored.
     */
    T *data = 0;
    /**
     * @brief the number of elements stored.
     */
    size_t length = 0;

    /**
     * @brief Release the elements stored
     */
    inline void ReleaseData()
    {
        if (data)
            delete[] data;
        data = 0;
        length = 0;
    }

    /**
     * @brief Allocate memory for the array.
     *
     * @param size the number of elements to store.
     */
    inline void AllocateData(size_t size)
    {
        length = size;
        data = length ? new T[length]{} : 0;
    }

    /**
     * @brief Copy elements from another array
     *
     * @param newData the pointer to the other array.
     * @param size the number of elements to copy.
     */
    inline void CopyData(const T *newData, size_t size)
    {
        for (size_t i = 0; i < size; i++)
            data[i] = newData[i];
    }

    /**
     * @brief Populate the array with a value
     *
     * @param element the value used to populate the array.
     * @param size the number of elements to store in the array.
     */
    inline void CopyData(const T &element, size_t size)
    {
        for (size_t i = 0; i < size; i++)
            data[i] = element;
    }

  public:
    /**
     * @brief type of the elements in the array.
     */
    using ValueType = T;

    /**
     * @brief Construct a new Array object
     */
    Array() noexcept
    {
    }

    /**
     * @brief Destroy the Array object
     */
    virtual ~Array() noexcept
    {
        ReleaseData();
    }

    /**
     * @brief Construct a new Array object
     *
     * @param initialSize the number of elements to store in the array.
     */
    Array(size_t initialSize) noexcept
    {
        AllocateData(initialSize);
    }

    /**
     * @brief Construct a new Array object
     *
     * @param arr the array to be copied into this array.
     */
    Array(const Array &arr) noexcept
    {
        AllocateData(arr.Length());
        CopyData(arr.data, arr.Length());
    }

    /**
     * @brief Construct a new Array object
     *
     * @param initList an initializer_list whose elements will be copied into this array.
     */
    Array(const std::initializer_list<T> &initList)
    {
        AllocateData(initList.size());
        size_t idx = 0;
        for (const auto &element : initList)
            data[idx++] = element;
    }

    /**
     * @brief Construct a new Array object
     *
     * @param arr the array to be 'moved' into this array.
     */
    Array(Array &&arr) noexcept
    {
        length = arr.Length();
        data = arr.data;
        arr.length = 0;
        arr.data = 0;
    }

    /**
     * @brief Construct a new Array object
     *
     * @param element the value used to populate the array.
     * @param nElements the number of elements the array will have.
     */
    Array(const T &element, size_t nElements) noexcept
    {
        AllocateData(nElements);
        CopyData(element, nElements);
    }

    /**
     * @brief Construct a new Array object
     *
     * @param cArr the pointer to the array to be copied into this array.
     * @param nElements the number of elements this array will have.
     */
    Array(const T *cArr, size_t nElements) noexcept
    {
        AllocateData(nElements);
        CopyData(cArr, nElements);
    }

    /**
     * @brief Array Copy Assignment
     *
     * @param arr the array to be copied into this array.
     * @return Array<T>& this array.
     */
    Array<T> &operator=(const Array<T> &arr) noexcept
    {
        ReleaseData();
        AllocateData(arr.Length());
        CopyData(arr.data, arr.Length());
        return *this;
    }

    /**
     * @brief Array Move Assignment
     *
     * @param arr the array to be 'moved' into this array.
     * @return Array<T>& this array.
     */
    Array<T> &operator=(Array<T> &&arr) noexcept
    {
        ReleaseData();
        length = arr.Length();
        data = arr.data;
        arr.length = 0;
        arr.data = 0;
        return *this;
    }

    /**
     * @brief Check if two arrays have the same number of equivalent elements
     *
     * @tparam Comparator the type of comparator that takes two elements and returns true if they are equivalent.
     * @param other another array.
     * @return true if both of the arrays have the same number of equivalent elements.
     * @return false otherwise.
     */
    template <class Comparator> bool operator==(const Array<T> &other) const noexcept
    {
        Comparator cmp;
        if (length != other.Length())
            return false;
        for (size_t i = 0; i < length; i++)
        {
            if (!cmp((*this)[i], other[i]))
                return false;
        }
        return true;
    }

    /**
     * @brief Get the pointer of the first element
     *
     * @return T* the pointer of the first element.
     */
    inline T *operator*() noexcept
    {
        return data;
    }

    /**
     * @brief Get the pointer of the first element
     *
     * @return const T* the pointer of the first element.
     */
    inline const T *operator*() const noexcept
    {
        return data;
    }

    /**
     * @brief Get the element at the given index
     *
     * @param index the index.
     * @return T& the element at the index.
     */
    inline T &operator[](size_t index) noexcept
    {
        return data[index];
    }

    /**
     * @brief Get the element at the given index
     *
     * @param index the index.
     * @return const T& the element at the index.
     */
    inline const T &operator[](size_t index) const noexcept
    {
        return data[index];
    }

    /**
     * @brief Get the number of elements in the array
     *
     * @return size_t the number of elements in the array.
     */
    inline virtual size_t Length() const noexcept
    {
        return length;
    }

    /**
     * @brief Get the first element in the array
     *
     * @return T& the first element in the array.
     */
    inline virtual T &GetFirst() noexcept
    {
        return data[0];
    }

    /**
     * @brief Get the first element in the array
     *
     * @return const T& the first element in the array.
     */
    inline virtual const T &GetFirst() const noexcept
    {
        return data[0];
    }

    /**
     * @brief Get the last element in the array
     *
     * @return T& the last element in the array.
     */
    inline virtual T &GetLast() noexcept
    {
        return data[length - 1];
    }

    /**
     * @brief Get the last element in the array
     *
     * @return const T& the last element in the array
     */
    inline virtual const T &GetLast() const noexcept
    {
        return data[length - 1];
    }
};
} // namespace ChiaData

#endif

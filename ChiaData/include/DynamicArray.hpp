#ifndef DYNAMIC_ARRAY_HPP
#define DYNAMIC_ARRAY_HPP

#include "Array.hpp"
#include "Types/Types.hpp"

namespace ChiaData
{
/**
 * @brief A data structure that is an dynamic array and expands its capacity when necessary.
 *
 * @tparam T the type of element.
 */
template <class T> class DynamicArray : public Array<T>
{
  private:
    /**
     * @brief the number of elements currently stored.
     */
    size_t nElements;

    inline void ResizeArray(size_t newSize) noexcept
    {
        auto oldData = this->data;
        Array<T>::AllocateData(newSize);
        for (size_t i = 0; i < nElements && i < newSize; i++)
            this->data[i] = oldData[i];
        if (oldData)
            delete[] oldData;
    }

    inline void DynamicallyResize() noexcept
    {
        if (nElements > this->length)
            ResizeArray(nElements);
        else if (nElements == this->length)
            ResizeArray(this->length > 2 ? this->length << 1 : 4);
        else if (this->length > 0 && nElements < (this->length >> 2))
            ResizeArray(this->length >> 1);
    }

  public:
    /**
     * @brief Construct a new empty DynamicArray object.
     */
    DynamicArray() noexcept : Array<T>(), nElements(0)
    {
    }

    /**
     * @brief Construct a new DynamicArray object with an initializer_list.
     *
     * @param initList an initializer_list with the elements to be stored in the DynamicArray.
     */
    DynamicArray(const std::initializer_list<T> &initList) noexcept : Array<T>(initList), nElements(initList.size())
    {
    }

    /**
     * @brief Construct a new DynamicArray object
     *
     * @param arr an Array to be copied to the DynamicArray.
     */
    DynamicArray(const Array<T> &arr) noexcept : Array<T>(arr)
    {
        nElements = arr.Length();
        if (auto dyArr = dynamic_cast<const DynamicArray<T> *>(&arr))
            nElements = dyArr->nElements;
    }

    /**
     * @brief Construct a new DynamicArray object
     *
     * @param arr an Array to be copied to the DynamicArray.
     */
    DynamicArray(Array<T> &&arr) noexcept
    {
        if (auto dyArr = dynamic_cast<DynamicArray<T> *>(&arr))
        {
            nElements = dyArr->nElements;
            dyArr->nElements = 0;
        }
        else
        {
            nElements = arr.length;
        }
        Array<T>::Array(Types::Forward<Array<T>>(arr));
    }

    /**
     * @brief Construct a new DynamicArray object of a given size.
     *
     * @param initialSize the size of the DynamicArray.
     */
    DynamicArray(size_t initialSize) noexcept : Array<T>(initialSize), nElements(initialSize)
    {
    }

    /**
     * @brief Construct a new DynamicArray object populated with the same elements.
     *
     * @param element the element to be copied.
     * @param nElements the number of copies.
     */
    DynamicArray(const T &element, size_t nElements) noexcept : Array<T>(element, nElements), nElements(nElements)
    {
    }

    /**
     * @brief Construct a new DynamicArray object
     *
     * @param cArr the array to be copied.
     * @param nElements the number of elements in the array.
     */
    DynamicArray(const T *cArr, size_t nElements) noexcept : Array<T>(cArr, nElements), nElements(nElements)
    {
    }

    /**
     * @brief Copy a std::initializer_list.
     *
     * @param l a std::initializer_list to be copied.
     * @return DynamicArray<T>& the DynamicArray.
     */
    DynamicArray<T> &operator=(const std::initializer_list<T> &l)
    {
        *this = DynamicArray<T>(l);
        return *this;
    }

    /**
     * @brief Copy an Array object.
     *
     * @param arr the Array object.
     * @return DynamicArray<T>& the DynamicArray.
     */
    DynamicArray<T> &operator=(const Array<T> &arr) noexcept
    {
        Array<T>::operator=(arr);
        if (auto dyArr = dynamic_cast<DynamicArray<T> *>(&arr))
            nElements = dyArr->nElements;
        else
            nElements = arr.length;
        return *this;
    }

    /**
     * @brief Move the content of an Array object.
     *
     * @param arr the Array object.
     * @return DynamicArray<T>& the DynamicArray.
     */
    DynamicArray<T> &operator=(Array<T> &&arr) noexcept
    {
        if (auto dyArr = dynamic_cast<DynamicArray<T> *>(&arr))
        {
            nElements = dyArr->nElements;
            dyArr->nElements = 0;
        }
        else
        {
            nElements = arr.Length();
        }
        Array<T>::operator=(Move(arr));
        return *this;
    }

    /**
     * @brief Check if each element in the DynamicArray is equal to the corresponding one in another DynamicArray.
     *
     * @tparam Comparator a type that checks if two elements are equal.
     * @param other the other DynamicArray.
     * @return true if each element in the DynamicArray is equal to the corresponding one in another DynamicArray.
     * @return false otherwise.
     */
    template <class Comparator> bool operator==(const Array<T> &other) const noexcept
    {
        Comparator cmp;
        size_t otherNElements = other.length;
        if (auto dyArr = dynamic_cast<DynamicArray<T> *>(&other))
        {
            otherNElements = dyArr->nElements;
        }
        if (nElements != otherNElements)
            return false;
        for (size_t i = 0; i < nElements; i++)
        {
            if (!cmp((*this)[i], other[i]))
                return false;
        }
        return true;
    }

    /**
     * @brief Append an element to the DynamicArray.
     *
     * @tparam Element the type of the element.
     * @param e the element to be appended.
     */
    template <class Element> inline void Append(Element &&e) noexcept
    {
        DynamicallyResize();
        this->data[nElements++] = Types::Forward<decltype(e)>(e);
    }

    /**
     * @brief Remove the last element out of the DynamicArray.
     */
    inline void RemoveLast() noexcept
    {
        if (IsEmpty())
            return;
        this->data[--nElements] = T();
        DynamicallyResize();
    }

    /**
     * @brief Remove all the elements out of the DynamicArray.
     */
    inline void RemoveAll() noexcept
    {
        nElements = 0;
        delete[] this->data;
        this->data = nullptr;
        DynamicallyResize();
    }

    /**
     * @brief Resize the DynamicArray.
     *
     * @param newSize the new size of the DynamicArray.
     */
    inline void Resize(size_t newSize) noexcept
    {
        ResizeArray(newSize);
        nElements = newSize;
    }

    /**
     * @brief Check if the DynamicArray is empty or not.
     *
     * @return true if it is empty.
     * @return false otherwise.
     */
    inline bool IsEmpty() const noexcept
    {
        return nElements == 0;
    }

    /**
     * @brief Return the number of elements in the DynamicArray.
     *
     * @return size_t the number of elements.
     */
    inline virtual size_t Length() const noexcept override
    {
        return nElements;
    }

    /**
     * @brief Get the first element out of the DynamicArray.
     *
     * @return T& the first element.
     */
    inline virtual T &GetFirst() noexcept override
    {
        return this->data[0];
    }

    /**
     * @brief Get the first element out of the DynamicArray.
     *
     * @return const T& the first element.
     */
    inline virtual const T &GetFirst() const noexcept override
    {
        return this->data[0];
    }

    /**
     * @brief Get the last element out of the DynamicArray.
     *
     * @return T& the last element.
     */
    inline virtual T &GetLast() noexcept override
    {
        return this->data[nElements - 1];
    }

    /**
     * @brief Get the last element out of the DynamicArray.
     *
     * @return const T& the last element.
     */
    inline virtual const T &GetLast() const noexcept override
    {
        return this->data[nElements - 1];
    }
};
} // namespace ChiaData

#endif

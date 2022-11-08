#ifndef CONTAINER_HPP
#define CONTAINER_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <array>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <vector>

#include "String.hpp"

namespace ChiaData
{
namespace Math
{
/**
 * @brief Container is an abstract class that is capable of storing numbers
 *
 * @tparam T the type of the elements.
 */
template <class T> class Container
{
  protected:
    // number of elements stored.
    size_t size;
    // array of the elements
    T *data;

  public:
    /**
     * @brief Construct a new Container object
     */
    Container() : size(0), data(nullptr)
    {
    }

    /**
     * @brief Construct a new Container object
     *
     * @param s the total number of elements to be stored.
     * @param value the value used to populate this container.
     */
    Container(size_t s, const T &value) : size(s), data(nullptr)
    {
        if (!s)
            return;
        data = new T[s];
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < s; i++)
            data[i] = value;
    }

    /**
     * @brief Construct a new Container object
     *
     * @param l an initializer_list whose elements will be copied to this container.
     */
    Container(const std::initializer_list<T> &l) : size(l.size()), data(nullptr)
    {
        if (!size)
            return;
        data = new T[size];
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < size; i++)
            data[i] = *(l.begin() + i);
    }

    /**
     * @brief Construct a new Container object
     *
     * @tparam N the size of the array.
     * @param arr the array whose elements will be copied to this container.
     */
    template <size_t N> Container(const std::array<T, N> &arr) : size(arr.size()), data(nullptr)
    {
        if (!size)
            return;
        data = new T[size];
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < size; i++)
            data[i] = arr[i];
    }

    /**
     * @brief Construct a new Container object
     *
     * @param values the vector whose elements will be copied to this container.
     */
    Container(const std::vector<T> &values) : size(values.size()), data(nullptr)
    {
        if (!size)
            return;
        data = new T[size];
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < size; i++)
            data[i] = values[i];
    }

    /**
     * @brief Construct a new Container object
     *
     * @tparam U the type of the elements of the other container.
     * @param other another container to be copied to this container.
     */
    template <class U> Container(const Container<U> &other) : Container(other.Size(), 0)
    {
        if (!size)
            return;
        data = new T[size];
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < size; i++)
            data[i] = other[i];
    }

    /**
     * @brief Construct a new Container object
     *
     * @param other another container to be 'moved' into this container.
     */
    Container(Container<T> &&other) noexcept : size(other.Size()), data(other.data)
    {
        other.size = 0;
        other.data = nullptr;
    }

    /**
     * @brief Destroy the Container object
     */
    virtual ~Container()
    {
        if (data)
            delete[] data;
    }

    /**
     * @brief Container Copy Assignment
     *
     * @tparam U the type of elements of the other container.
     * @param other a container to be copied to this container.
     * @return Container<T>& this container.
     */
    template <class U> Container<T> &operator=(const Container<U> &other)
    {
        if (this == &other)
            return *this;

        size = other.Size();
        if (data)
            delete[] data;
        data = nullptr;
        if (size > 0)
        {
            data = new T[size];
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < size; i++)
                data[i] = T(other[i]);
        }
        return *this;
    }

    /**
     * @brief Container Move Assignment
     *
     * @param other another container whose elements will be 'moved' into this container.
     * @return Container<T>& this container.
     */
    virtual Container<T> &operator=(Container<T> &&other) noexcept
    {
        if (this == &other)
            return *this;
        this->size = other.size;
        this->data = other.data;
        other.size = 0;
        other.data = nullptr;
        return *this;
    }

    /**
     * @brief Get the size (number of elements) of the container
     *
     * @return size_t the number of elements this container stores.
     */
    virtual size_t Size() const
    {
        return size;
    }

    /**
     * @brief Check if this container is empty or not
     *
     * @return true if the container does not have any element.
     * @return false if the container has at least one element.
     */
    virtual bool IsEmpty() const
    {
        return !size;
    }

    /**
     * @brief Generate a string that describes the container
     *
     * @return String a string that describes the container.
     */
    virtual String ToString() const = 0;

    /**
     * @brief Generate a string that describes the container
     *
     * @return std::string a string that describes the container.
     */
    virtual std::string ToStdString() const = 0;

    template <class U> friend class Container;
};

/**
 * @brief Convert the container to a string and pass it to a string stream
 *
 * @tparam T the type of elements in the container.
 * @param stream the string stream.
 * @param container the container
 * @return StringStream& the string stream.
 */
template <class T> StringStream &operator<<(StringStream &stream, const Container<T> &container)
{
    stream << container.ToString();
    return stream;
}

/**
 * @brief Convert the container to a string and pass it to a string stream
 *
 * @tparam T the type of elements in the container.
 * @param stream the string stream.
 * @param container the container.
 * @return std::stringstream& the string stream.
 */
template <class T> std::stringstream &operator<<(std::stringstream &stream, const Container<T> &container)
{
    stream << container.ToStdString();
    return stream;
}

/**
 * @brief Convert the container to a string and pass it to an output stream.
 *
 * @tparam T the type of the elements in the container.
 * @param stream the output stream.
 * @param container the container
 * @return std::ostream& the output stream.
 */
template <class T> std::ostream &operator<<(std::ostream &stream, const Container<T> &container)
{
    stream << container.ToStdString();
    return stream;
}
} // namespace Math
} // namespace ChiaData

#endif // CONTAINER_HPP

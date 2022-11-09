#ifndef TUPLE_HPP
#define TUPLE_HPP

#include "Math/Container.hpp"
#include "Types/Types.hpp"

namespace ChiaData
{
namespace Math
{
/**
 * @brief Tuple is an immutable container of a fixed size
 * 
 * @tparam T the type of the elements.
 */
template <class T> class Tuple : public Container<T>
{
  public:
    /**
     * @brief Construct a new Tuple object
     */
    Tuple() : Container<T>()
    {
    }

    /**
     * @brief Construct a new Tuple object
     *
     * @param s the size (number of elements) of the tuple.
     * @param value the value used to populate the tuple.
     */
    Tuple(size_t s, const T &value) : Container<T>(s, value)
    {
    }

    /**
     * @brief Construct a new Tuple object
     *
     * @param l an initializer_list whose elements will be copied to the tuple.
     */
    Tuple(const std::initializer_list<T> &l) : Container<T>(l)
    {
    }

    /**
     * @brief Construct a new Tuple object
     *
     * @tparam N the size of the array.
     * @param arr an array whose elements will be copied to this tuple.
     */
    template <size_t N> Tuple(const std::array<T, N> &arr) : Container<T>(arr)
    {
    }

    /**
     * @brief Construct a new Tuple object
     *
     * @param values a vector whose elements will be copied to this tuple.
     */
    Tuple(const std::vector<T> &values) : Container<T>(values)
    {
    }

    /**
     * @brief Construct a new Tuple object
     *
     * @tparam U the type of elements in the container.
     * @param other a container whose elements will be copied to this tuple.
     */
    template <class U> Tuple(const Container<U> &other) : Container<T>(other)
    {
    }

    /**
     * @brief Construct a new Tuple object
     *
     * @param other a container whose elements will be copied to this tuple.
     */
    Tuple(Container<T> &&other) : Container<T>(Types::Move(other))
    {
    }

    /**
     * @brief Access the element at a given index
     *
     * @param index the index used to access the element in this tuple.
     * @return const T& the element at the index.
     * @throw IndexOutOfBound if the index exceeds the largest possible index.
     */
    virtual const T &operator[](const size_t &index) const
    {
        if (index < this->size)
            return this->data[index];
        StringStream stream;
        stream << "Tuple - Index Out of Bound:\n";
        stream << "Size: " << Size() << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Tuple Copy Assignment
     *
     * @tparam U the type of elements in the container.
     * @param other a container whose elements will be copied to this tuple.
     * @return Tuple<T>& the tuple.
     */
    template <class U> Tuple<T> &operator=(const Container<U> &other)
    {
        Container<T>::operator=(other);
        return *this;
    }

    /**
     * @brief Tuple Move Assignment
     *
     * @param other a container whose elements will be copied to this tuple.
     * @return Tuple<T>& the tuple.
     */
    virtual Tuple<T> &operator=(Container<T> &&other) noexcept override
    {
        Container<T>::operator=(Types::Move(other));
        return *this;
    }

    /**
     * Check if two Tuples have the same elements
     * @param other a Tuple to be compared with this Tuple.
     * @return a bool that indicates whether both Tuples have the same elements.
     **/

    /**
     * @brief Check if two tuples have the equivalent elements
     *
     * @tparam U the type of elements the other tuple has
     * @param other a tuple to be compared with this tuple.
     * @return true if two tuples have the equivalent elements.
     * @return false if two tuples have different elements.
     */
    template <class U> bool operator==(const Tuple<U> &other) const
    {
        // Check if the same Tuple is being compared.
        if (this == &other)
            return true;
        // Both must have the same size.
        if (this->Size() != other.Size())
            return false;
        // Check if each pair of elements have identical values.
        for (size_t i = 0; i < this->Size(); i++)
            if ((*this)[i] != other[i])
                return false;
        return true;
    }

    /**
     * @brief Check if two tuples have different elements
     *
     * @tparam U the type of elements of the other tuple.
     * @param other a tuple to be compared with this tuple.
     * @return true if the tuples have different elements.
     * @return false if the tuples have the equivalent elements.
     */
    template <class U> bool operator!=(const Tuple<U> &other) const
    {
        return !operator==(other);
    }

    /**
     * @brief Generate a string that describes the elements of tuple
     *
     * @return String a string that describes the elements of tuple.
     */
    virtual String ToString() const override
    {
        StringStream stream;
        stream << "(";
        for (size_t i = 0; i < Size(); i++)
        {
            stream << (*this)[i];
            if (i < Size() - 1)
                stream << ", ";
        }
        stream << ")";
        return stream.ToString();
    }

    /**
     * @brief Generate a string that describes the elements of tuple
     *
     * @return std::string a string that describes the elements of tuple.
     */
    virtual std::string ToStdString() const override
    {
        std::stringstream ss;
        ss << "(";
        for (size_t i = 0; i < Size(); i++)
        {
            ss << (*this)[i];
            if (i < Size() - 1)
                ss << ", ";
        }
        ss << ")";
        return ss.str();
    }

    template <class OtherType> friend class Tuple;
};
} // namespace Math
} // namespace ChiaData

#endif

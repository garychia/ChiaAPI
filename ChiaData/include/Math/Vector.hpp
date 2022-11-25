#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "Math.hpp"
#include "Tuple.hpp"

namespace ChiaData
{
namespace Math
{
/**
 * @brief Vector is a mutable container of a fixed size that supports numerical operations
 *
 * @tparam T the type of the elements.
 */
template <class T> class Vector : public Tuple<T>
{
  public:
    /**
     * @brief Construct a new Vector object
     */
    Vector() : Tuple<T>()
    {
    }

    /**
     * @brief Construct a new Vector object
     *
     * @param s the size of the vector.
     * @param value the value used to populate the vector.
     */
    Vector(size_t s, const T &value = 0) : Tuple<T>(s, value)
    {
    }

    /**
     * @brief Construct a new Vector object
     *
     * @param l an initializer_list whose elements will be copied to this vector.
     */
    Vector(const std::initializer_list<T> &l) : Tuple<T>(l)
    {
    }

    /**
     * @brief Construct a new Vector object
     *
     * @tparam N the size of the array.
     * @param arr an array whose elements will be copied into this vector.
     */
    template <size_t N> Vector(const std::array<T, N> &arr) : Tuple<T>(arr)
    {
    }

    /**
     * @brief Construct a new Vector object
     *
     * @tparam U the type of elements of the given vector.
     * @param v a vector those elements will be copied to this vector.
     */
    template <class U> Vector(const std::vector<U> &v) : Tuple<T>(v)
    {
    }

    /**
     * @brief Construct a new Vector object
     *
     * @tparam U the type of elements of the tuple.
     * @param other a tuple whose elements will be copied to this vector.
     */
    template <class U> Vector(const Tuple<U> &other) : Tuple<T>(other)
    {
    }

    /**
     * @brief Construct a new Vector object
     *
     * @param other a tuple whose elements will be 'moved' into this vector.
     */
    Vector(Tuple<T> &&other) : Tuple<T>(Types::Move(other))
    {
    }

    /**
     * @brief Vector Copy Assignment
     *
     * @tparam U the type of elements of the given tuple.
     * @param other a tuple whose elements will be copied to this vector.
     * @return Vector<T>& this vector.
     */
    template <class U> Vector<T> &operator=(const Tuple<U> &other)
    {
        Tuple<T>::operator=(other);
        return *this;
    }

    /**
     * @brief Vector Move Assignment
     *
     * @param other a tuple whose elements will be 'moved' into this vector.
     * @return Vector<T>& this vector.
     */
    Vector<T> &operator=(Tuple<T> &&other)
    {
        Tuple<T>::operator=(other);
        return *this;
    }

    /**
     * @brief Access the element at a given index.
     *
     * @param index the index at which the element will be accessed.
     * @return T& the element at the index.
     * @throw IndexOutOfBound if the index exceeds the largest possible index.
     */
    virtual T &operator[](const size_t &index)
    {
        if (index < Tuple<T>::Size())
            return this->data[index];
        StringStream stream;
        stream << "Vector - Index Out of Bound:\n";
        stream << "Size: " << Tuple<T>::Size() << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Access the element at a given index.
     *
     * @param index the index at which the element will be accessed.
     * @return T& the element at the index.
     * @throw IndexOutOfBound if the index exceeds the largest possible index.
     **/
    virtual const T &operator[](const size_t &index) const override
    {
        if (index < Tuple<T>::Size())
            return this->data[index];
        StringStream stream;
        stream << "Vector - Index Out of Bound:\n";
        stream << "Size: " << Tuple<T>::Size() << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Return the dimension (number of elements) of vector.
     *
     * @return size_t the dimension.
     */
    size_t Dimension() const
    {
        return Tuple<T>::Size();
    }

    /**
     * @brief Compute the Euclidean norm of the vector
     *
     * @tparam ReturnType the type of the result.
     * @return ReturnType the Euclidean norm of this vector.
     */
    template <class ReturnType> ReturnType Length() const
    {
        return LpNorm<ReturnType>(2);
    }

    /**
     * @brief Return the Euclidean norm of the Vector (Same as Vector::Length)
     *
     * @tparam ReturnType the type of the output.
     * @return ReturnType the Euclidean norm of this vector.
     */
    template <class ReturnType> ReturnType EuclideanNorm() const
    {
        return Length<ReturnType>();
    }

    /**
     * @brief Compute the Lp norm of the vector
     *
     * @tparam ReturnType the type of the output.
     * @param p the dimension.
     * @return ReturnType the Lp norm.
     */
    template <class ReturnType> ReturnType LpNorm(int p) const
    {
        ReturnType squaredTotal = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : squaredTotal)
        for (size_t i = 0; i < Dimension(); i++)
            squaredTotal += ChiaMath::template Power<ReturnType, int>((*this)[i], p);
        return ChiaMath::template Power<ReturnType, double>(squaredTotal, (double)1 / p);
    }

    /**
     * @brief Perform addition with two vectors
     *
     * @tparam U the type of the other vector.
     * @param other a vector to be added to this vector.
     * @return auto the result of the addition.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> auto Add(const Vector<U> &other) const
    {
        Vector<decltype((*this)[0] + other[0])> result(*this);
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return result;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream << "Vector::Add: the argument is expected to have a dimension that is a factor of that of this "
                      "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            result[i] += other[i % other.Dimension()];
        return result;
    }

    /**
     * @brief Perform element-wise addition with a vector and a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be added to each element of the vector.
     * @return auto the result vector.
     */
    template <class ScalerType> auto Add(const ScalerType &scaler) const
    {
        Vector<decltype((*this)[0] + scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            result[i] += scaler;
        return result;
    }

    /**
     * @brief Perform addition with two Vectors (Same as Vector::Add)
     *
     * @tparam U the type of the elements of the other vector.
     * @param other a vector to be added to this vector.
     * @return auto the result vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> auto operator+(const Vector<U> &other) const
    {
        return this->Add(other);
    }

    /**
     * @brief Perform element-wise addition with a vector and a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be added to each element of the vector.
     * @return auto the result vector.
     */
    template <class ScalerType> auto operator+(const ScalerType &scaler) const
    {
        return this->Add(scaler);
    }

    /**
     * @brief Perform inplace addition with another vector
     *
     * @tparam U the type of elements in the other vector.
     * @param other a vector to be added to this vector.
     * @return Vector<T>& this vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> Vector<T> &operator+=(const Vector<U> &other)
    {
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return *this;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream
                << "Vector::operator+=: the argument is expected to have a dimension that is a factor of that of this "
                   "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            this->data[i] += other[i % other.Dimension()];
        return *this;
    }

    /**
     * @brief Perform inplace addition with a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be added to this vector.
     * @return Vector<T>& this vector.
     */
    template <class ScalerType> Vector<T> &operator+=(const ScalerType &scaler)
    {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            this->data[i] += scaler;
        return *this;
    }

    /**
     * @brief Perform subtraction with two vectors
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector that is subtracted from this vector.
     * @return auto the result vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> auto Minus(const Vector<U> &other) const
    {
        Vector<decltype((*this)[0] - other[0])> result(*this);
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return result;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream << "Vector::Minus: the argument is expected to have a dimension that is a factor of that of this "
                      "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            result[i] -= other[i % other.Dimension()];
        return result;
    }

    /**
     * @brief Perform element-wise subtraction with a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be subtracted from each element of this vector.
     * @return auto the result vector.
     */
    template <class ScalerType> auto Minus(const ScalerType &scaler) const
    {
        Vector<decltype(this->data[0] - scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            result[i] -= scaler;
        return result;
    }

    /**
     * @brief Perform subtraction with two Vectors (Same as Vector::Minus)
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector to be subtracted from this vector.
     * @return auto the reuslt vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> auto operator-(const Vector<U> &other) const
    {
        return this->Minus(other);
    }

    /**
     * @brief Perform subtraction with a vector and a scaler (Same as Vector.Minus)
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be subtracted from each element of this vector.
     * @return auto the result vector.
     */
    template <class ScalerType> auto operator-(const ScalerType &scaler) const
    {
        return this->Minus(scaler);
    }

    /**
     * @brief Perform inplace subtraction with another vector
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector to be subtracted from this vector.
     * @return Vector<T>& this vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> Vector<T> &operator-=(const Vector<U> &other)
    {
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return *this;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream
                << "Vector::operator-=: the argument is expected to have a dimension that is a factor of that of this "
                   "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            this->data[i] -= other[i % other.Dimension()];
        return *this;
    }

    /**
     * @brief Perform inplace subtraction with a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be subtracted from each element of this vector.
     * @return Vector<T>& this vector.
     */
    template <class ScalerType> Vector<T> &operator-=(const ScalerType &scaler)
    {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            this->data[i] -= scaler;
        return *this;
    }

    /**
     * @brief Perform element-wise multiplication with a vector and a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be multiplied with each element of this vector.
     * @return auto the result vector.
     */
    template <class ScalerType> auto Scale(const ScalerType &scaler) const
    {
        Vector<decltype(this->data[0] * scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            result[i] *= scaler;
        return result;
    }

    /**
     * @brief Perform element-wise multiplication with two vectors
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector where each of its elements will be multiplied with the corresponding element(s) of this
     * vector.
     * @return auto the result vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> auto Scale(const Vector<U> &other) const
    {
        Vector<decltype((*this)[0] * other[0])> result(*this);
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return result;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream << "Vector::Scale: the argument is expected to have a dimension that is a factor of that of this "
                      "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            result[i] *= other[i % other.Dimension()];
        return result;
    }

    /**
     * @brief Perform element-wise multiplication with a Vector and a scaler (Same as Vector::Scale)
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be multiplied with each element of this vector.
     * @return auto the result vector.
     */
    template <class ScalerType> auto operator*(const ScalerType &scaler) const
    {
        return this->Scale(scaler);
    }

    /**
     * @brief Perform element-wise multiplication with two vectors
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector where each of its element will be multiplied with the corresponding element(s) of this
     * vector.
     * @return auto the result vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> auto operator*(const Vector<U> &other) const
    {
        Vector<decltype((*this)[0] * other[0])> result(*this);
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return result;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream
                << "Vector::operator*: the argument is expected to have a dimension that is a factor of that of this "
                   "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            result[i] *= other[i % other.Dimension()];
        return result;
    }

    /**
     * @brief Perform inplace element-wise multiplication with a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be multiplied with each element of this vector.
     * @return Vector<T>& this vector.
     */
    template <class ScalerType> Vector<T> &operator*=(const ScalerType &scaler)
    {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            (*this)[i] *= scaler;
        return *this;
    }

    /**
     * @brief Perform inplace element-wise multiplication with a vector
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector where each of its element will be multiplied with the corresponding element(s) of this
     * vector.
     * @return Vector<T>& this vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     */
    template <class U> Vector<T> &operator*=(const Vector<U> &other)
    {
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return *this;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream
                << "Vector::operator*=: the argument is expected to have a dimension that is a factor of that of this "
                   "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            (*this)[i] *= other[i % other.Dimension()];
        return *this;
    }

    /**
     * @brief Divide each element of the vector by a scaler.
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler used to divide each element of this vector.
     * @return auto the reusult vector.
     * @throw DividedByZero if the scaler is 0.
     */
    template <class ScalerType> auto Divide(const ScalerType &scaler) const
    {
        if (scaler == 0)
        {
            StringStream stream;
            stream << "Vector - Divided by Zero:\n";
            stream << "Vector::Divide: the argument is expected to be non-zero.\n";
            throw ChiaRuntime::DividedByZero(stream.ToString());
        }
        Vector<decltype((*this)[0] / scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            result[i] /= scaler;
        return result;
    }

    /**
     * @brief Perform element-wise division with two vectors
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector where each of its elements will divide the corresponding element(s) of this vector.
     * @return auto the result vector.
     * @throw InvalidArgument if the argument does not have a dimension that is a factor of that of this vector.
     * @throw DividedByZero if at least one of the elements of argument is 0.
     */
    template <class U> auto Divide(const Vector<U> &other) const
    {
        Vector<decltype((*this)[0] / other[0])> result(*this);
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return result;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream << "Vector::Divide: the argument is expected to have a dimension that is a factor of that of this "
                      "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }

        for (size_t i = 0; i < other.Dimension(); i++)
        {
            if (other[i] == 0)
            {
                StringStream stream;
                stream << "Vector - Divided by Zero:\n";
                stream << "Vector::Divide: the value at index " << i << " in the argument is found to be zero.";
                throw ChiaRuntime::DividedByZero(stream.ToString());
            }
        }

        size_t j;
#pragma omp parallel for schedule(dynamic) private(j)
        for (size_t i = 0; i < Dimension(); i++)
        {
            j = i % other.Dimension();
            result[i] /= other[j];
        }
        return result;
    }

    /**
     * @brief Divide each element of a vector by a scaler (Same as Vector::Divide)
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler used to divide each element of this vector.
     * @return auto the result.
     * @throw DividedByZero if the scaler is 0.
     */
    template <class ScalerType> auto operator/(const ScalerType &scaler) const
    {
        return this->Divide(scaler);
    }

    /**
     * @brief Perform element-wise division with two vectors (Same as Vector::Divide)
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector where each of its elements will divide the corresponding element(s) of this vector.
     * @return auto the result.
     * @throw DividedByZero if at least one of the elements of argument is 0.
     */
    template <class U> auto operator/(const Vector<U> &other) const
    {
        return this->Divide(other);
    }

    /**
     * @brief Perform inplace element-wise division with a vector and a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler used to divide each element of this vector.
     * @return Vector<T>& this vector.
     * @throw DividedByZero if the scaler is 0.
     */
    template <class ScalerType> Vector<T> &operator/=(const ScalerType &scaler)
    {
        if (scaler == 0)
        {
            StringStream stream;
            stream << "Vector - Divided by Zero:\n";
            stream << "Vector::operator/=: the argument is expected to be non-zero.\n";
            throw ChiaRuntime::DividedByZero(stream.ToString());
        }

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
            (*this)[i] /= scaler;
        return *this;
    }

    /**
     * @brief Perform inplace element-wise division
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector where each of its elements will divide the corresponding element(s) of this vector.
     * @return Vector<T>& this vector.
     * @throw DividedByZero if at least one of the elements of argument is 0.
     */
    template <class U> Vector<T> &operator/=(const Vector<U> &other)
    {
        if (Tuple<T>::IsEmpty() || other.IsEmpty())
            return *this;
        else if (Dimension() % other.Dimension() != 0)
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream
                << "Vector::operator/=: the argument is expected to have a dimension that is a factor of that of this "
                   "vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Dimension(); i++)
        {
            const auto element = other[i % other.Dimension()];
            if (element == 0)
                throw ChiaRuntime::DividedByZero("Vector: Expect none of the element of the second operand to be 0 "
                                                 "when performing"
                                                 "element-wise division.");
            (*this)[i] /= element;
        }
        return *this;
    }

    /**
     * @brief Perform dot product with two vectors.
     *
     * @tparam U the type of elements of the other vector.
     * @param other a vector.
     * @return auto the result.
     * @throw InvalidArgument if the dimension of the argument is not equal to that of this vector.
     */
    template <class U> auto Dot(const Vector<U> &other) const
    {
        if (Dimension() != other.Dimension())
        {
            StringStream stream;
            stream << "Vector - Invalid Argument:\n";
            stream << "Vector::Dot: the dimension of the argument is expected to be the same as that of this vector.\n";
            stream << "Dimension of this vector: " << Dimension() << "\n";
            stream << "Dimension of the argument: " << other.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
        return Scale(other).Sum();
    }

    /**
     * @brief Compute the normalized vector.
     *
     * @tparam ReturnType the type of elements of the result vector.
     * @return Vector<ReturnType> the normalized vector.
     * @throw DividedByZero if the summation of the elements of this vector is zero.
     */
    template <class ReturnType> Vector<ReturnType> Normalized() const
    {
        const ReturnType length = Length<ReturnType>();
        if (length == 0)
        {
            StringStream stream;
            stream << "Vector - Divided by Zero:\n";
            stream << "Vector::Normalized: the summation of elements of the vector is expected to be non-zero.\n";
            throw ChiaRuntime::DividedByZero(stream.ToString());
        }
        return *this / length;
    }

    /**
     * @brief Normalize the vector inplace
     *
     * @return Vector<T>& this vector.
     * @throw DividedByZero if the summation of the elements of this vector is zero.
     */
    Vector<T> &Normalize()
    {
        const T length = Length<T>();
        if (length == 0)
        {
            StringStream stream;
            stream << "Vector - Divided by Zero:\n";
            stream << "Vector::Normalize: the summation of elements of the vector is expected to be non-zero.\n";
            throw ChiaRuntime::DividedByZero(stream.ToString());
        }
        *this /= length;
    }

    /**
     * @brief Compute the summation of elements of the vector.
     *
     * @return T the summation of the elements.
     */
    T Sum() const
    {
        T total = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : total)
        for (size_t i = 0; i < Container<T>::Size(); i++)
            total += (*this)[i];
        return total;
    }

    /**
     * @brief Map each element of the vector to a new value
     *
     * @tparam MapFunction the type of function that maps each element of this vector to a new value.
     * @param f a function that maps each element of this vector to a new value.
     * @return auto the result vector after applying the function to the elements.
     */
    template <class MapFunction> auto Map(MapFunction &&f) const
    {
        Vector<decltype(f((*this)[0]))> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < Tuple<T>::Size(); i++)
            result[i] = f(result[i]);
        return result;
    }

    /**
     * @brief Return a pointer that points to the first element of the vector
     *
     * @return const T* the pointer.
     */
    const T *AsRawPointer() const
    {
        return this->data;
    }

    /**
     * @brief Construct a new vector filled with zeros
     *
     * @param n the number of zeros (dimension of the vector).
     * @return Vector<T> the result vector.
     */
    static Vector<T> ZeroVector(const size_t &n)
    {
        return Vector<T>(n, 0);
    }

    /**
     * @brief Pack all the elements of multiple vectors into a new vector
     *
     * @param vectors an initializer_list of vectors to be packed.
     * @return Vector<T> a new vector with all of the elements packed.
     */
    static Vector<T> Combine(const std::initializer_list<Vector<T>> &vectors)
    {
        size_t elementTotal = 0;
        for (auto itr = vectors.begin(); itr != vectors.end(); itr++)
        {
            elementTotal += itr->Size();
        }
        Vector<T> combined(elementTotal, 0);
        size_t currentIndex = 0;
        for (auto vector : vectors)
        {
            for (size_t j = 0; j < vector.Size(); j++)
                combined[currentIndex++] = vector[j];
        }
        return combined;
    }

    /**
     * @brief Perform addition with a scaler and a vector
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be added to each element of the vector.
     * @param v a vector to be added.
     * @return auto the result.
     */
    template <class ScalerType> friend auto operator+(const ScalerType &scaler, const Vector<T> &v)
    {
        Vector<decltype(scaler + v[0])> result(v);
#pragma omp parallel for
        for (size_t i = 0; i < result.Dimension(); i++)
            result[i] += scaler;
        return result;
    }

    /**
     * @brief Perform subtraction with a scaler and a vector
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler used to perform element-wise subtraction with a scaler and a vector.
     * @param v a vector.
     * @return auto the result vector.
     */
    template <class ScalerType> friend auto operator-(const ScalerType &scaler, const Vector<T> &v)
    {
        Vector<decltype(scaler - v[0])> result(v);
#pragma omp parallel for
        for (size_t i = 0; i < result.Dimension(); i++)
            result[i] = scaler - result[i];
        return result;
    }

    /**
     * @brief Perform element-wise multiplication with a scaler and a vector
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be multiplied with each element of the vector.
     * @param v a vector.
     * @return auto the result vector.
     */
    template <class ScalerType> friend auto operator*(const ScalerType &scaler, const Vector<T> &v)
    {
        Vector<decltype(scaler * v[0])> result(v);
#pragma omp parallel for
        for (size_t i = 0; i < result.Dimension(); i++)
            result[i] *= scaler;
        return result;
    }

    /**
     * @brief Perform element-wise division with a scaler and a vector
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler to be divided by each element of the vector.
     * @param v a vector.
     * @return auto the result vector.
     * @throw DividedByZero if at least one of the elements of the vector is 0.
     */
    template <class ScalerType> friend auto operator/(const ScalerType &scaler, const Vector<T> &v)
    {
        for (size_t i = 0; i < v.Dimension(); i++)
        {
            if (!v[i])
            {
                StringStream stream;
                stream << "Vector - Divided by Zero:\n";
                stream << "Vector::operator/: the value at index " << i << " in the argument is found to be zero.";
                throw ChiaRuntime::DividedByZero(stream.ToString());
            }
        }
        Vector<decltype(scaler / v[0])> result(v);
#pragma omp parallel for
        for (size_t i = 0; i < result.Dimension(); i++)
            result[i] = scaler / result[i];
        return result;
    }

    template <class U> friend class Vector;
};
} // namespace Math
} // namespace ChiaData

#endif

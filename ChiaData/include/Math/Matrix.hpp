#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Array.hpp"
#include "Vector.hpp"

namespace ChiaData
{
namespace Math
{
/**
 * @brief Matrix is a data structure that has rows and columns
 *
 * @tparam T the type of elements in the matrix.
 */
template <class T> class Matrix : Container<T>
{
  private:
    // number of rows
    size_t nRows;
    // number of columns
    size_t nColumns;

  public:
    class Row
    {
      private:
        // pointer to the owner matrix
        Matrix *pMatrix;
        // index of row
        size_t row;

        Row(Matrix *pMatrix, size_t row) : pMatrix(pMatrix), row(row)
        {
        }

      public:
        /**
         * @brief Get the element at the given column index
         *
         * @param index the column index.
         * @return T& the element at the index.
         */
        T &operator[](size_t index)
        {
            Tuple<T> matrixShape = pMatrix->Shape();
            if (index < matrixShape[1])
                return pMatrix->GetElement(row, index);
            StringStream stream;
            stream << "Matrix::Row - Index Out of Bound:\n";
            stream << "Number of columns: " << matrixShape[1] << "\n";
            stream << "Index: " << index << "\n";
            throw ChiaRuntime::IndexOutOfBound(stream.ToString());
        }

        /**
         * @brief Get the row index of this row
         *
         * @return size_t the row index.
         */
        size_t GetRowIndex() const
        {
            return row;
        }

        friend class Matrix<T>;
    };

    class ConstRow
    {
      private:
        // pointer to the owner matrix
        const Matrix *pMatrix;
        // index of row
        size_t row;

        ConstRow(const Matrix *pMatrix, size_t row) : pMatrix(pMatrix), row(row)
        {
        }

      public:
        /**
         * @brief Get the element at the given column index
         *
         * @param index the column index.
         * @return const T& the element at the index.
         */
        const T &operator[](size_t index) const
        {
            Tuple<T> matrixShape = pMatrix->Shape();
            if (index < matrixShape[1])
                return pMatrix->GetElement(row, index);
            StringStream stream;
            stream << "Matrix::ConstRow - Index Out of Bound:\n";
            stream << "Number of columns: " << matrixShape[1] << "\n";
            stream << "Index: " << index << "\n";
            throw ChiaRuntime::IndexOutOfBound(stream.ToString());
        }

        /**
         * @brief Get the row index of this row
         *
         * @return size_t the row index.
         */
        size_t GetRowIndex() const
        {
            return row;
        }

        friend class Matrix<T>;
    };

    /**
     * @brief Construct a new Matrix object
     */
    Matrix() : Container<T>(), nRows(0), nColumns(0)
    {
    }

    /**
     * @brief Construct a new Matrix object
     *
     * @param nRows the number of rows in the matrix.
     * @param nColumns the number of columns in the matrix.
     * @param initialValue the value used to populate the matrix.
     */
    Matrix(size_t nRows, size_t nColumns, const T &initialValue = 0)
        : Container<T>(nRows * nColumns, initialValue), nRows(nRows), nColumns(nColumns)
    {
    }

    /**
     * @brief Construct a new Matrix object
     *
     * @param l an initializer_list of row vectors to be packed into this Matrix.
     * @throw InvalidArgument if all of the row vectors do not have the same dimension.
     */
    Matrix(const std::initializer_list<Vector<T>> &l)
        : Container<T>(l.size() ? l.size() * l.begin()->Size() : 0, 0), nRows(l.size()),
          nColumns(l.size() ? l.begin()->Size() : 0)
    {
        for (const Vector<T> &vector : l)
        {
            if (vector.Dimension() != nColumns)
            {
                StringStream stream;
                stream << "Matrix - Invalid Argument:\n";
                stream << "Matrix::Matrix: each of the vectors in the initializer_list is expected to have the same "
                          "dimension.\n";
                throw ChiaRuntime::InvalidArgument(stream.ToString());
            }
        }

        auto itr = l.begin();
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                (*this)[i][j] = (*(itr + i))[j];
            }
        }
    }

    /**
     * @brief Construct a new Matrix object
     *
     * @param l an initializer_list of values to be packed into this matrix as a row or column vector.
     * @param column If true, the resulting matrix will be a column vector. Otherwise, it will be a row vector.
     */
    Matrix(const std::initializer_list<T> &l, bool column = true) : Container<T>(l.size(), 0)
    {
        nRows = column ? l.size() : 1;
        nColumns = column ? 1 : l.size();
        auto itr = l.begin();
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < l.size(); i++)
        {
            if (column)
                (*this)[i][0] = *(itr + i);
            else
                (*this)[0][i] = *(itr + i);
        }
    }

    /**
     * @brief Construct a new Matrix object
     *
     * @tparam N the size of the array.
     * @param arr an array of row vectors that will be copied into this matrix.
     * @throw InvalidArgument if all of the row vectors do not have the same dimension.
     */
    template <size_t N>
    Matrix(const std::array<Vector<T>, N> &arr)
        : Container<T>(arr.size() ? arr.size() * arr[0].Size() : 0, 0), nRows(arr.size()), nColumns(arr.size() ? arr[0].Size())
    {
        for (const Vector<T> &vector : arr)
        {
            if (vector.Dimension() != nColumns)
            {
                StringStream stream;
                stream << "Matrix - Invalid Argument:\n";
                stream << "Matrix::Matrix: each of the vectors in the array is expected to have the same dimension.\n";
                throw ChiaRuntime::InvalidArgument(stream.ToString());
            }
        }
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
                (*this)[i][j] = arr[i][j];
        }
    }

    /**
     * @brief Construct a new Matrix object
     *
     * @param arr an array of row vectors that will be packed into this matrix.
     * @throw InvalidArgument if all of the row vectors do not have the same dimension.
     */
    Matrix(const Array<Vector<T>> &arr)
        : Container<T>(arr.Length() ? arr.Length() * arr[0].Dimension() : 0, 0), nRows(arr.Length()),
          nColumns(arr.Length() ? arr[0].Dimension() : 0)
    {
        size_t i, j;
        for (i = 0; i < arr.Length(); i++)
        {
            if (arr[i].Dimension() != nColumns)
            {
                StringStream stream;
                stream << "Matrix - Invalid Argument:\n";
                stream << "Matrix::Matrix: each of the vectors in the array is expected to have the same dimension.\n";
                throw ChiaRuntime::InvalidArgument(stream.ToString());
            }
        }
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                (*this)[i][j] = arr[i][j];
            }
        }
    }

    /**
     * @brief Construct a new Matrix object
     *
     * @param other a matrix to be copied into this matrix.
     */
    Matrix(const Matrix<T> &other) : Container<T>(other), nRows(other.nRows), nColumns(other.nColumns)
    {
    }

    /**
     * @brief Construct a new Matrix object
     *
     * @tparam U the type of elements of the other matrix.
     * @param other a matrix that will be copied to this matrix.
     */
    template <class U>
    Matrix(const Matrix<U> &other) : Container<T>(other), nRows(other.nRows), nColumns(other.nColumns)
    {
    }

    /**
     * @brief Construct a new Matrix object
     *
     * @param other a matrix that will be 'moved' into this matrix.
     */
    Matrix(Matrix<T> &&other) noexcept : Container<T>(Types::Move(other)), nRows(other.nRows), nColumns(other.nColumns)
    {
        other.nRows = 0;
        other.nColumns = 0;
    }

    /**
     * @brief Matrix Copy Assignment
     *
     * @param other a matrix to be copied into this matrix.
     * @return Matrix<T>& this matrix.
     */
    Matrix<T> &operator=(const Matrix<T> &other)
    {
        if (this == &other)
            return *this;
        Container<T>::operator=(other);
        nRows = other.nRows;
        nColumns = other.nColumns;
        return *this;
    }

    /**
     * @brief Matrix Copy Assignment
     *
     * @tparam U the type of the elements of the other matrix.
     * @param other a matrix that will be copied to this matrix.
     * @return Matrix<T>& this matrix.
     */
    template <class U> Matrix<T> &operator=(const Matrix<U> &other)
    {
        if (this == &other)
            return *this;
        Container<T>::operator=(other);
        nRows = other.nRows;
        nColumns = other.nColumns;
        return *this;
    }

    /**
     * @brief Matrix Move Assignment
     *
     * @param other a matrix that will be 'moved' into this matrix.
     * @return Matrix<T>& this matrix.
     */
    virtual Matrix<T> &operator=(Matrix<T> &&other) noexcept
    {
        if (this == &other)
            return *this;
        Container<T>::operator=(Types::Move(other));
        nRows = other.nRows;
        nColumns = other.nColumns;
        other.nRows = 0;
        other.nColumns = 0;
        return *this;
    }

    /**
     * @brief Access the row at a given index
     *
     * @param index the index of the row to be accessed.
     * @return Row the row at the index.
     * @throw IndexOutOfBound when the index exceeds the greatest possible index.
     */
    virtual Row operator[](size_t index)
    {
        if (index < nRows)
            return Row(this, index);
        StringStream stream;
        stream << "Matrix - Index Out of Bound:\n";
        stream << "Matrix::operator[]: the row index is expected to be less than the number of rows.\n";
        stream << "Number of rows: " << nRows << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Access the row at a given index
     *
     * @param index the index of the row to be accessed.
     * @return ConstRow the row at the index.
     * @throw IndexOutOfBound when the index exceeds the greatest possible index.
     */
    virtual ConstRow operator[](size_t index) const
    {
        ConstRow row(this, index);
        if (index < nRows)
            return row;
        StringStream stream;
        stream << "Matrix - Index Out of Bound:\n";
        stream << "Matrix::operator[]: the row index is expected to be less than the number of rows.\n";
        stream << "Number of rows: " << nRows << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Get the element at the given index in row major order
     *
     * @param index the index at which the element will be accessed.
     * @return T& the element.
     * @throw IndexOutOfBound when the index exceeds the greatest possible index.
     */
    virtual T &GetElement(size_t index)
    {
        if (index < Size())
            return this->data[index];
        StringStream stream;
        stream << "Matrix - Index Out of Bound:\n";
        stream << "Matrix::GetElement: the index is expected to be less than the total number of elements in the "
                  "matrix.\n";
        stream << "Number of rows: " << nRows << "\n";
        stream << "Number of columns: " << nColumns << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Get the element at the given row and column indices.
     *
     * @param row the row index.
     * @param column the column index.
     * @return T& the element.
     */
    virtual T &GetElement(size_t row, size_t column)
    {
        const size_t index = row * nColumns + column;
        if (index < Size())
            return this->data[index];
        StringStream stream;
        stream << "Matrix - Index Out of Bound:\n";
        stream << "Matrix::GetElement: the index is expected to be less than the total number of elements in the "
                  "matrix.\n";
        stream << "Number of rows: " << nRows << "\n";
        stream << "Number of columns: " << nColumns << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Get the element at the given index in row major order
     *
     * @param index the index at which the element will be accessed.
     * @return const T& the element.
     * @throw IndexOutOfBound when the index exceeds the greatest possible index.
     */
    virtual const T &GetElement(size_t index) const
    {
        if (index < Size())
            return this->data[index];
        StringStream stream;
        stream << "Matrix - Index Out of Bound:\n";
        stream << "Matrix::GetElement: the index is expected to be less than the total number of elements in the "
                  "matrix.\n";
        stream << "Number of rows: " << nRows << "\n";
        stream << "Number of columns: " << nColumns << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Get the element at the given row and column indices.
     *
     * @param row the row index.
     * @param column the column index.
     * @return const T& the element.
     */
    virtual const T &GetElement(size_t row, size_t column) const
    {
        const size_t index = row * nColumns + column;
        if (index < Size())
            return this->data[index];
        StringStream stream;
        stream << "Matrix - Index Out of Bound:\n";
        stream << "Matrix::GetElement: the index is expected to be less than the total number of elements in the "
                  "matrix.\n";
        stream << "Number of rows: " << nRows << "\n";
        stream << "Number of columns: " << nColumns << "\n";
        stream << "Index: " << index << "\n";
        throw ChiaRuntime::IndexOutOfBound(stream.ToString());
    }

    /**
     * @brief Get the shape of the matrix
     *
     * @return Tuple<size_t> a tuple of size 2 that contains the number of rows and that of columns.
     */
    virtual Tuple<size_t> Shape() const
    {
        return Tuple<size_t>({nRows, nColumns});
    }

    /**
     * @brief Perform addition with two matrices
     *
     * @tparam U the type of elements in the other matrix.
     * @param other the other matrix.
     * @return auto the result matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     */
    template <class U> auto Add(const Matrix<U> &other) const
    {
        Matrix<decltype((*this)[0][0] + other[0][0])> result(*this);
        if (IsEmpty() || other.IsEmpty())
            return result;
        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        if (!(thisShape[0] % otherShape[0] == 0 && thisShape[1] % otherShape[1] == 0))
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream << "Matrix::Add: the numbers of rows and columns in the argument are expected to be a factor of "
                      "these of rows and columns respectively in this matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                result.GetElement(i, j) += other.GetElement(i % other.nRows, j % other.nColumns);
            }
        }
        return result;
    }

    /**
     * @brief Perform addition with two matrices (Same as Matrix::Add)
     *
     * @tparam U the type of elements in the other matrix.
     * @param other the other matrix.
     * @return auto the result matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     */
    template <class U> auto operator+(const Matrix<U> &other) const
    {
        return Add(other);
    }

    /**
     * @brief Perform inplace matrix addition with another matrix
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return Matrix<T>& this matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     */
    template <class U> Matrix<T> &operator+=(const Matrix<U> &other)
    {
        if (IsEmpty() || other.IsEmpty())
            return *this;
        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        if (!(thisShape[0] % otherShape[0] == 0 && thisShape[1] % otherShape[1] == 0))
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream
                << "Matrix::operator+=: the numbers of rows and columns in the argument are expected to be a factor of "
                   "these of rows and columns respectively in this matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                (*this).GetElement(i, j) += other.GetElement(i % other.nRows, j % other.nColumns);
            }
        }
        return *this;
    }

    /**
     * @brief Perform subtraction with two matrices
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return auto the result matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     */
    template <class U> auto Subtract(const Matrix<U> &other) const
    {
        Matrix<decltype((*this)[0][0] + other[0][0])> result(*this);
        if (IsEmpty() || other.IsEmpty())
            return result;
        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        if (!(thisShape[0] % otherShape[0] == 0 && thisShape[1] % otherShape[1] == 0))
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream
                << "Matrix::Subtract: the numbers of rows and columns in the argument are expected to be a factor of "
                   "these of rows and columns respectively in this matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }

        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                result.GetElement(i, j) -= other.GetElement(i % other.nRows, j % other.nColumns);
            }
        }
        return result;
    }

    /**
     * @brief Perform subtraction with two matrices (Same as Matrix::Subtract)
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return auto the result matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     */
    template <class U> auto operator-(const Matrix<U> &other) const
    {
        return Subtract(other);
    }

    /**
     * @brief Perform inplace subtraction with another matrix
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix
     * @return Matrix<T>& this matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     */
    template <class U> Matrix<T> &operator-=(const Matrix<U> &other)
    {
        if (IsEmpty() || other.IsEmpty())
            return *this;
        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        if (!(thisShape[0] % otherShape[0] == 0 && thisShape[1] % otherShape[1] == 0))
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream
                << "Matrix::operator-=: the numbers of rows and columns in the argument are expected to be a factor of "
                   "these of rows and columns respectively in this matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                (*this).GetElement(i, j) -= other.GetElement(i % other.nRows, j % other.nColumns);
            }
        }
        return *this;
    }

    /**
     * @brief Perform matrix multiplication with two matrices
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return auto the result.
     * @throw InvalidArgument if the number of columns in this matrix is not the same as that of rows in the argument
     * matrix.
     */
    template <class U> auto Multiply(const Matrix<U> &other) const
    {
        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        Matrix<decltype((*this)[0][0] * other[0][0])> result(thisShape[0], otherShape[1]);
        if (IsEmpty() || other.IsEmpty())
            return result;
        if (thisShape[1] != otherShape[0])
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream << "Matrix::Multiply: the number of columns of this matrix is expected to be the same as that of "
                      "rows in the argument matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }

        size_t i, j, k;
#pragma omp parallel for private(k, j) schedule(dynamic) collapse(3)
        for (i = 0; i < thisShape[0]; i++)
            for (k = 0; k < thisShape[1]; k++)
                for (j = 0; j < otherShape[1]; j++)
#pragma omp atomic
                    result.GetElement(i, j) += (*this).GetElement(i, k) * other.GetElement(k, j);
        return result;
    }

    /**
     * @brief Perform matrix multiplication with two matrices (Same as Matrix::Multiply)
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return auto the result.
     * @throw InvalidArgument if the number of columns in this matrix is not the same as that of rows in the argument
     * matrix.
     */
    template <class U> auto operator*(const Matrix<U> &other) const
    {
        return Multiply(other);
    }

    /**
     * @brief Perform element-wise multiplication with a matrix and scaler
     *
     * @param scaler a scaler.
     * @return auto the result matrix.
     */
    auto Scale(const T &scaler) const
    {
        Matrix<decltype((*this)[0][0] * scaler)> result(*this);
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                result[i][j] *= scaler;
            }
        }
        return result;
    }

    /**
     * @brief Perform element-wise multiplication with two matrices
     *
     * @tparam U the type of elements in the other matrix
     * @param other a matrix
     * @return auto the result matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     */
    template <class U> auto Scale(const Matrix<U> &other) const
    {
        Matrix<decltype((*this)[0][0] * other[0][0])> result(*this);
        if (IsEmpty() || other.IsEmpty())
            return result;

        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        if (thisShape[0] % otherShape[0] != 0 || thisShape[1] % otherShape[1] != 0)
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream << "Matrix::Scale: the numbers of rows and columns in the argument are expected to be a factor of "
                      "these of rows and columns respectively in this matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }

        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                result.GetElement(i, j) *= other.GetElement(i % other.nRows, j % other.nColumns);
            }
        }
        return result;
    }

    /**
     * @brief Perform element-wise multiplication with a matrix and scaler (Same as Matrix::Scale)
     *
     * @param scaler a scaler.
     * @return auto the result matrix.
     */
    auto operator*(const T &scaler) const
    {
        return Scale(scaler);
    }

    /**
     * @brief Perform inplace element-wise multiplication with a scaler
     *
     * @param scaler a scaler.
     * @return Matrix<T>& this matrix.
     */
    Matrix<T> &operator*=(const T &scaler)
    {
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
                (*this).GetElement(i, j) *= scaler;
        }
        return *this;
    }

    /**
     * @brief Perform inplace element-wise multiplication with another matrix
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return Matrix<T>& this matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     */
    template <class U> Matrix<T> &operator*=(const Matrix<U> &other)
    {
        if (IsEmpty() || other.IsEmpty())
            return *this;

        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        if (thisShape[0] % otherShape[0] != 0 || thisShape[1] % otherShape[1] != 0)
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream
                << "Matrix::operator*=: the numbers of rows and columns in the argument are expected to be a factor of "
                   "these of rows and columns respectively in this matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }

        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
                (*this).GetElement(i, j) *= other.GetElement(i % other.nRows, j % other.nColumns);
        }
        return *this;
    }

    /**
     * @brief Perform element-wise division with a matrix and a scaler
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler.
     * @return auto the result matrix.
     * @throw DividedByZero if the scaler is zero.
     */
    template <class ScalerType> auto Divide(const ScalerType &scaler) const
    {
        if (scaler == 0)
        {
            StringStream stream;
            stream << "Matrix - Divided by Zero:\n";
            stream << "Matrix::Divide: the scaler argument is expected to be non-zero.\n";
            throw ChiaRuntime::DividedByZero(stream.ToString());
        }

        Matrix<decltype((*this)[0][0] / scaler)> result(*this);
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
                result.GetElement(i, j) /= scaler;
        }
        return result;
    }

    /**
     * @brief Perform element-wise division with a matrix and a scaler
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return auto the result matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     * @throw DividedByZero if at least one of the elements in the other matrix is zero.
     */
    template <class U> auto Divide(const Matrix<U> &other) const
    {
        Matrix<decltype((*this)[0][0] / other[0][0])> result(*this);
        if (IsEmpty() || other.IsEmpty())
            return result;

        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        if (thisShape[0] % otherShape[0] != 0 || thisShape[1] % otherShape[1] != 0)
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream << "Matrix::Divide: the numbers of rows and columns in the argument are expected to be a factor of "
                      "these of rows and columns respectively in this matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }

        for (size_t i = 0; i < other.nRows; i++)
        {
            for (size_t j = 0; j < other.nColumns; j++)
            {
                if (other.GetElement(i, j) == 0)
                {
                    StringStream stream;
                    stream << "Matrix - Divided by Zero:\n";
                    stream << "Matrix::Divide: all of the elements of the argument are expected to be non-zero.\n";
                    stream << "Row index of the argument (i =): " << i << "\n";
                    stream << "Column index of the argument: (j =)" << j << "\n";
                    stream << "argument[i][j] = " << other.GetElement(i, j) << "\n";
                    throw ChiaRuntime::DividedByZero(stream.ToString());
                }
            }
        }

        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
                result.GetElement(i, j) /= other.GetElement(i % other.nRows, j % other.nColumns);
        }
        return result;
    }

    /**
     * @brief Perform element-wise division with a scaler (Same as Matrix::Divide)
     *
     * @tparam ScalerType the type of the scaler.
     * @param scaler a scaler.
     * @return auto the result matrix.
     * @throw DividedByZero if the scaler is zero.
     */
    template <class ScalerType> auto operator/(const ScalerType &scaler) const
    {
        return Divide(scaler);
    }

    /**
     * @brief Performs inplace element-wise division with a scaler
     *
     * @param scaler a scaler.
     * @return Matrix<T>& this matrix.
     * @throw DividedByZero if the scaler is zero.
     */
    Matrix<T> &operator/=(const T &scaler)
    {
        if (scaler == 0)
        {
            StringStream stream;
            stream << "Matrix - Divided by Zero:\n";
            stream << "Matrix::operator/=: the scaler argument is expected to be non-zero.\n";
            throw ChiaRuntime::DividedByZero(stream.ToString());
        }

        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
                (*this).GetElement(i, j) /= scaler;
        }
        return *this;
    }

    /**
     * @brief Perform inplace element-wise division with another matrix
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return Matrix<T>& this matrix.
     * @throw InvalidArgument the shape of the other matrix is not a factor of that of this matrix.
     * @throw DividedByZero if at least one of the elements in the other matrix is zero.
     */
    template <class U> Matrix<T> &operator/=(const Matrix<U> &other)
    {
        if (IsEmpty() || other.IsEmpty())
            return *this;

        const auto thisShape = Shape();
        const auto otherShape = other.Shape();
        if (thisShape[0] % otherShape[0] != 0 || thisShape[1] % otherShape[1] != 0)
        {
            StringStream stream;
            stream << "Matrix - Invalid Argument\n";
            stream << "Matrix::Scale: the numbers of rows and columns in the argument are expected to be a factor of "
                      "these of rows and columns respectively in this matrix.\n";
            stream << "Shape of this matrix: " << thisShape.ToString() << "\n";
            stream << "Shape of the argument: " << otherShape.ToString() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.ToString());
        }

        for (size_t i = 0; i < other.nRows; i++)
        {
            for (size_t j = 0; j < other.nColumns; j++)
            {
                if (other.GetElement(i, j) == 0)
                {
                    StringStream stream;
                    stream << "Matrix - Divided by Zero:\n";
                    stream << "Matrix::Divide: all of the elements of the argument are expected to be non-zero.\n";
                    stream << "Row index of the argument (i =): " << i << "\n";
                    stream << "Column index of the argument: (j =)" << j << "\n";
                    stream << "argument[i][j] = " << other.GetElement(i, j) << "\n";
                    throw ChiaRuntime::DividedByZero(stream.ToString());
                }
            }
        }

        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
                (*this).GetElement(i, j) /= other.GetElement(i % other.nRows, j % other.nColumns);
        }
        return *this;
    }

    /**
     * @brief Check if two matrices have the same shape and elements with the equivalent values
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return true if two matrices have the same shape and elements with the equivalent values.
     * @return false otherwise.
     */
    template <class U> bool operator==(const Matrix<U> &other) const
    {
        if (!(nRows == other.nRows && nColumns == other.nColumns))
            return false;
        for (size_t i = 0; i < Size(); i++)
        {
            if (GetElement(i) != other.GetElement(i))
                return false;
        }
        return true;
    }

    /**
     * @brief Check if two matrices have different shapes or at least one element in one matrix has a value different
     * from the corresponding one in the other matrix.
     *
     * @tparam U the type of elements in the other matrix.
     * @param other a matrix.
     * @return true if two matrices have the same shape and elements with the equivalent values.
     * @return false otherwise.
     */
    template <class U> bool operator!=(const Matrix<U> &other) const
    {
        return !operator==(other);
    }

    /**
     * @brief Generate a string that describes the matrix
     *
     * @return String a string that describes the matrix.
     */
    virtual String ToString() const override
    {
        StringStream stream;
        stream << "[";
        for (size_t i = 0; i < nRows; i++)
        {
            stream << (i == 0 ? "[" : " [");
            for (size_t j = 0; j < nColumns; j++)
            {
                stream << (*this).GetElement(i, j);
                if (j < nColumns - 1)
                    stream << ", ";
            }
            stream << "]";
            if (i < nRows - 1)
                stream << ","
                       << "\n";
        }
        stream << "]";
        return stream.ToString();
    }

    /**
     * @brief Generate a string that describes the matrix
     *
     * @return std::string a string that describes the matrix.
     */
    virtual std::string ToStdString() const override
    {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < nRows; i++)
        {
            ss << (i == 0 ? "[" : " [");
            for (size_t j = 0; j < nColumns; j++)
            {
                ss << (*this).GetElement(i, j);
                if (j < nColumns - 1)
                    ss << ", ";
            }
            ss << "]";
            if (i < nRows - 1)
                ss << "," << std::endl;
        }
        ss << "]";
        return ss.str();
    }

    /**
     * @brief Transpose the matrix inplace
     */
    void Transpose()
    {
        if (IsEmpty())
            return;
        T *newElements = new T[Size()];
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < nRows; i++)
        {
            for (j = 0; j < nColumns; j++)
            {
                newElements[j * nRows + i] = (*this).GetElement(i, j);
            }
        }
        delete this->data;
        this->data = newElements;
        auto temp = nRows;
        nRows = nColumns;
        nColumns = temp;
    }

    /**
     * @brief Compute the transpose of the matrix
     *
     * @return Matrix<T> the transpose.
     */
    Matrix<T> Transposed() const
    {
        Matrix<T> result(*this);
        result.Transpose();
        return result;
    }

    /*
    Constructs a new Matrix by flattening this Matrix in row-major or
    column-major order.
    @param rowMajor true if flattening in row-major. False if flattening
    in column-major order.
    @param keepInRow true if all the elements will be placed in a
    single row. False if they will be placed in a single column.
    @return a Matrix with a single row or column.
    */

    /**
     * @brief Generate a new matrix by flattening a matrix in row-major or column-major order.
     *
     * @param rowMajor a bool indicates if the matrix will be flatten in the row-major manner.
     * @param keepInRow a bool indicates if the elements will be kept in a single row.
     * @return Matrix<T> the result matrix.
     */
    Matrix<T> Flattened(bool rowMajor = true, bool keepInRow = true) const
    {
        Matrix<T> result(*this);
        if (!rowMajor)
            result.Transpose();
        const auto nElements = result.nRows * result.nColumns;
        result.nColumns = keepInRow ? nElements : 1;
        result.nRows = keepInRow ? 1 : nElements;
        return result;
    }

    /**
     * @brief Compute the summation of all the elements in the matrix
     *
     * @return T the summation.
     */
    T SumAll() const
    {
        T result = 0;
        size_t i;
#pragma omp parallel for schedule(dynamic) reduction(+ : sum)
        for (i = 0; i < Size(); i++)
            result += GetElement(i);
        return result;
    }

    /**
     * @brief Calculate the summation of all the rows or columns of the Matrix
     *
     * @param sumRows a bool indicates if the summation will be performed on the rows.
     * @return Matrix<T> the result matrix.
     */
    Matrix<T> Sum(bool sumRows = true) const
    {
        Matrix<T> result(sumRows ? 1 : nRows, sumRows ? nColumns : 1);
        size_t i, j;
        if (sumRows)
        {
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
            for (i = 0; i < nRows; i++)
            {
                for (j = 0; j < nColumns; j++)
                    result.GetElement(j) += (*this).GetElement(i, j);
            }
        }
        else
        {
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
            for (i = 0; i < nRows; i++)
            {
                for (j = 0; j < nColumns; j++)
                    result.GetElement(i, 0) += (*this).GetElement(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Calculate the Frobenius norm of the matrix.
     *
     * @tparam ReturnType the type of the output.
     * @return ReturnType the Frobenius norm.
     */
    template <class ReturnType> ReturnType FrobeniusNorm() const
    {
        return Math::Power<T, double>(Map([](const T &e) { return e * e; }).SumAll(), 0.5);
    }

    /**
     * @brief Maps each element of the matrix to a new value
     *
     * @tparam MapFunction the type of the mapping function
     * @param f a function that maps each element of this matrix to a new value.
     * @return auto the result of the mapping.
     */
    template <class MapFunction> auto Map(MapFunction &&f) const
    {
        Matrix<T> result(*this);
        size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
        for (i = 0; i < result.nRows; i++)
        {
            for (j = 0; j < result.nColumns; j++)
                result.GetElement(i, j) = f(result.GetElement(i, j));
        }
        return result;
    }

    /**
     * @brief Construct an identity matrix
     *
     * @param n the number of rows (columns) of the matrix.
     * @return Matrix<T> the identity matrix.
     */
    static Matrix<T> Identity(size_t n)
    {
        Matrix<T> result(n, n);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n; i++)
            result.GetElement(i, i) = 1;
        return result;
    }

    /**
     * @brief Construct a diagonal matrix.
     *
     * @param values a vector whose elements will be the diagonal entries of the diagonal matrix.
     * @return Matrix<T> the diagonal matrix.
     */
    static Matrix<T> Diagonal(const Vector<T> &values)
    {
        const auto n = values.Dimension();
        auto result = Identity(n, n);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n; i++)
            result.GetElement(i, i) = values[i];
        return result;
    }

    /**
     * @brief Construct a translation matrix
     *
     * @param deltas a translation vector.
     * @return Matrix<T> the translation matrix.
     */
    static Matrix<T> Translation(const Vector<T> &deltas)
    {
        const size_t n = deltas.Dimension() + 1;
        auto result = Identity(n);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n - 1; i++)
            result.GetElement(i, n - 1) = deltas[i];
        return result;
    }

    /**
     * @brief Construct a scaling matrix
     *
     * @param factors a vector with the factors on each axis.
     * @return Matrix<T> the scaling matrix.
     */
    static Matrix<T> Scaling(const Vector<T> &factors)
    {
        const size_t n = factors.Dimension() + 1;
        auto result = Identity(n);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n - 1; i++)
            result.GetElement(i, i) *= factors[i];
        return result;
    }

    /**
     * @brief Construct a rotation matrix in 2D space.
     *
     * @param radians the angle in radians of the rotation.
     * @return Matrix<T> the rotation matrix.
     */
    static Matrix<T> Rotation2D(const T &radians)
    {
        const T sinValue = Math::Sine(radians);
        const T cosValue = Math::Cosine(radians);
        return Matrix<T>({{cosValue, -sinValue, 0}, {sinValue, cosValue, 0}, {0, 0, 1}});
    }

    /**
     * @brief Construct a rotation matrix in 3D space
     *
     * @param axis the rotation axis in 3D space.
     * @param radians the angle in radians of the rotation.
     * @return Matrix<T> the rotation matrix.
     * @throw InvalidArgument if the axis does not have a dimension of 3.
     */
    static Matrix<T> Rotation3D(const Vector<T> &axis, const T &radians)
    {
        if (axis.Dimension() != 3)
        {
            StringStream stream;
            stream << "Matrix - InvalidArgument:\n";
            stream << "Matrix::Rotation3D: the rotation axis is expected to have a dimension of 3.\n";
            stream << "Dimension of the axis: " << axis.Dimension() << "\n";
            throw ChiaRuntime::InvalidArgument(stream.str());
        }
        const auto normalizedAxis = axis.Normalized();
        const T &x = normalizedAxis[0];
        const T &y = normalizedAxis[1];
        const T &z = normalizedAxis[2];
        const T sinValue = Math::Sine(radians);
        const T cosValue = Math::Cosine(radians);
        const T oneMinusCosValue = 1 - cosValue;
        return Matrix<T>({{cosValue + x * x * oneMinusCosValue, x * y * oneMinusCosValue - z * sinValue,
                           x * z * oneMinusCosValue + y * sinValue, 0},
                          {y * x * oneMinusCosValue + z * sinValue, cosValue + y * y * oneMinusCosValue,
                           y * z * oneMinusCosValue - x * sinValue, 0},
                          {z * x * oneMinusCosValue - y * sinValue, z * y * oneMinusCosValue + x * sinValue,
                           cosValue + z * z * oneMinusCosValue, 0},
                          {0, 0, 0, 1}});
    }

    /**
     * @brief Construct a perspective projection matrix
     *
     * @param fov the field of view in radians.
     * @param aspect the aspect ratio (ratio of width and height of the viewport).
     * @param near the distance from the camera to the near plane.
     * @param far the distance from the camera to the far plane.
     * @return Matrix<T> the projection matrix.
     * @throw InvalidArgument if the field of view is not positive or the aspect ratio is zero.
     */
    static Matrix<T> Perspective(T fov, T aspect, T near, T far)
    {
        if (fov <= 0)
        {
            StringStream stream;
            stream << "Matrix - InvalidArgument:\n";
            stream << "Matrix::Perspective: the field of view is expected to be positive.\n";
            stream << "Field of View (FOV): " << fov << "\n";
            throw ChiaRuntime::InvalidArgument(stream.str());
        }
        if (aspect == 0)
        {
            StringStream stream;
            stream << "Matrix - InvalidArgument:\n";
            stream << "Matrix::Perspective: the aspect ratio is expected to be non-zero.\n";
            throw ChiaRuntime::InvalidArgument(stream.str());
        }
        const T scale = 1 / Math::Tangent(fov * 0.5);
        const T farNearDiff = far - near;
        return Matrix<T>({{scale * aspect, 0, 0, 0},
                          {0, scale, 0, 0},
                          {0, 0, -(far + near) / farNearDiff, -2 * near * far / farNearDiff},
                          {0, 0, -1, 0}});
    }

    /**
     * @brief Construct an othographic projection matrix
     *
     * @param left the horizontal coordinate of the left of the frustum.
     * @param right the horizontal coordinate of the right of the frustum.
     * @param bottom the vertical coordinate of the bottom of the frustum.
     * @param top the vertical coordinate of the top of the frustum.
     * @param near the distance from the camera to the near plane.
     * @param far the distance from the camera to the far plane.
     * @return Matrix<T> the othographic projection matrix.
     * @throw InvalidArgument if 'left' is equal to 'right', 'bottom' is equal to 'top', or 'near' is equal to 'far'.
     */
    static Matrix<T> Orthographic(T left, T right, T bottom, T top, T near, T far)
    {
        if (left == right)
        {
            StringStream stream;
            stream << "Matrix - InvalidArgument:\n";
            stream << "Matrix::Orthographic: 'left' and 'right' are expected to be different.\n";
            stream << "left = right = " << left << "\n";
            throw ChiaRuntime::InvalidArgument(stream.str());
        }
        if (bottom == top)
        {
            StringStream stream;
            stream << "Matrix - InvalidArgument:\n";
            stream << "Matrix::Orthographic: 'bottom' and 'top' are expected to be different.\n";
            stream << "bottom = top = " << bottom << "\n";
            throw ChiaRuntime::InvalidArgument(stream.str());
        }
        if (near == far)
        {
            StringStream stream;
            stream << "Matrix - InvalidArgument:\n";
            stream << "Matrix::Orthographic: 'near' and 'far' are expected to be different.\n";
            stream << "near = far = " << near << "\n";
            throw ChiaRuntime::InvalidArgument(stream.str());
        }
        const T rightLeftDiff = right - left;
        const T topBottomDiff = top - bottom;
        const T farNearDist = far - near;
        return Matrix<T>({{2 / rightLeftDiff, 0, 0, -(right + left) / rightLeftDiff},
                          {0, 2 / topBottomDiff, 0, -(top + bottom) / topBottomDiff},
                          {0, 0, -2 / farNearDist, -(far + near) / farNearDist},
                          {0, 0, 0, 1}});
    }

    /**
     * @brief Perform element-wise multiplication with a scaler and a matrix.
     *
     * @param scaler a scaler.
     * @param matrix a matrix.
     * @return auto the result matrix.
     */
    friend auto operator*(const double &scaler, const Matrix<T> &matrix)
    {
        return matrix.Map([&scaler](T e) { return scaler * e; });
    }

    /**
     * @brief Perform element-wise division with a scaler and a matrix.
     *
     * @param scaler a scaler.
     * @param matrix a matrix.
     * @return auto the result matrix.
     */
    friend auto operator/(const double &scaler, const Matrix<T> &matrix)
    {
        return matrix.Map([&scaler](T e) { return scaler / e; });
    }

    template <class U> friend class Matrix;
};

/**
 * @brief Convert the matrix to a string and pass it to the string stream.
 *
 * @tparam T the type of elements in the matrix.
 * @param stream the string stream.
 * @param m the matrix.
 * @return StringStream& the string stream.
 */
template <class T> StringStream &operator<<(StringStream &stream, const Matrix<T> &m)
{
    stream << m.ToString();
    return stream;
}

/**
 * @brief Convert the matrix to a string and pass it to the output stream.
 *
 * @tparam T the type of elements in the matrix.
 * @param os the output stream.
 * @param m the matrix.
 * @return std::ostream& the output stream.
 */
template <class T> std::ostream &operator<<(std::ostream &os, const Matrix<T> &m)
{
    os << m.ToStdString();
    return os;
}
} // namespace Math
} // namespace ChiaData

#endif

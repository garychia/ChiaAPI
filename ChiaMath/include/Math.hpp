#ifndef MATH_HPP
#define MATH_HPP

namespace ChiaData
{
namespace Math
{
template <class T> class List;
template <class T> class Matrix;
template <class T> class Vector;
} // namespace Math
} // namespace ChiaData

namespace ChiaMath
{
class Constants
{
  public:
    static const double Epsilon;
    static const double Ln2;
    static const double Pi;
};

/**
 * @brief Calculate the absolute value of a given value
 *
 * @tparam T the type of the input parameter.
 * @param x the input value.
 * @return T the absolute value.
 */
template <class T> T Abs(const T &x);

/**
 * @brief Get the greater of two given values
 *
 * @tparam T the type of the input values
 * @param x the first value.
 * @param y the second value.
 * @return T& the value that is greater than or equal to the other.
 */
template <class T> T &Max(const T &x, const T &y);

/**
 * @brief Get the less of two given values
 *
 * @tparam T the type of the input values.
 * @param x the first value.
 * @param y the second value.
 * @return T& the value is less than or equal to the other.
 */
template <class T> T &Min(const T &x, const T &y);

/**
 * @brief Calculate the value of Euler's number (e) raised to the power of a given value
 *
 * @tparam T the type of the input value.
 * @param x the exponent.
 * @return T the exponential to the power of x.
 */
template <class T> T Exp(const T &x);

/**
 * @brief Compute the natural logarithm of a given value
 *
 * @tparam T the type of the input.
 * @param x the input to the natural logarithm function.
 * @return T the natural logarithm of x.
 */
template <class T> T NaturalLog(const T &x);

/**
 * @brief Compute the sine of a given value
 *
 * @tparam T the type of the input.
 * @param x the input value in radians to the sine function.
 * @return T the sine of x.
 */
template <class T> T Sine(const T &x);

/**
 * @brief Compute the cosine of a given value
 *
 * @tparam T the type of the input.
 * @param x the input value in radians to the cosine function.
 * @return T the cosine of x.
 */
template <class T> T Cosine(const T &x);

/**
 * @brief Compute the tangent of a given value
 *
 * @tparam T the type of the input.
 * @param x the input value in radians to the tangent function.
 * @return T the tangent of x.
 */
template <class T> T Tangent(const T &x);

/**
 * @brief Compute the hyperbolic sine of a given value
 *
 * @tparam T the type of the input.
 * @param x the input value in radians to the hyperbolic sine function.
 * @return T the hyperbolic sine of x.
 */
template <class T> T Sinh(const T &x);

/**
 * @brief Compute the hyperbolic cosine of a given value
 *
 * @tparam T the type of the input.
 * @param x the input value in radians to the hyperbolic cosine function.
 * @return T the hyperbolic cosine of x.
 */
template <class T> T Cosh(const T &x);

/**
 * @brief Compute the hyperbolic tangent of a given value
 *
 * @tparam T the type of the input.
 * @param x the input value in radians to the hyperbolic tangent function.
 * @return T the hyperbolic tangent of x.
 */
template <class T> T Tanh(const T &x);

/**
 * @brief Calculate a given value raised to the power of a given exponent
 *
 * @tparam T the type of the base value.
 * @tparam PowerType the type of the exponent.
 * @param x the base value.
 * @param n the exponent.
 * @return T the value of x raised to the power of n.
 */
template <class T, class PowerType> T Power(const T x, PowerType n);

/**
 * @brief Rectified Linear Unit Function
 *
 * @tparam T the type of the input.
 * @param x the input to the ReLU function.
 * @return T the output of the function given x.
 */
template <class T> T ReLU(const T &x);

/**
 * @brief Sigmoid Function
 *
 * @tparam T the type of the input.
 * @param x the input to the sigmoid function.
 * @return T the output of the sigmoid function given x.
 */
template <class T> T Sigmoid(const T &x);

/**
 * @brief Softmax Funciton
 *
 * @tparam T the type of the elements of the given vector.
 * @param vector the input vector to the softmax function.
 * @return ChiaData::Vector<T> the output of the softmax function given the input vector.
 */
template <class T> ChiaData::Math::Vector<T> Softmax(const ChiaData::Math::Vector<T> &vector);

/**
 * @brief Softmax Function
 *
 * @tparam T the type of the elements of the input matrix.
 * @param matrix the input matrix to the softmax function.
 * @return ChiaData::Matrix<T> the output matrix with each of its column applied to by the softmax function.
 */
template <class T> ChiaData::Math::Matrix<T> Softmax(const ChiaData::Math::Matrix<T> &matrix);

/**
 * @brief Gaussian Probability Density Function
 *
 * @tparam T the type of the input, mean, and standard deviation values.
 * @param x the input to the function.
 * @param mu the mean (average).
 * @param sigma the standard deviation.
 * @return T the output of the Gaussian Probability Density function given x, mu and sigma.
 */
template <class T> T Gauss(const T &x, const T &mu, const T &sigma);
} // namespace ChiaMath

#include "Exceptions.hpp"
#include "List.hpp"
#include "Math/Matrix.hpp"
#include "Math/Vector.hpp"
#include "String.hpp"

namespace ChiaMath
{
template <class T> T Abs(const T &x)
{
    return x >= 0 ? x : -x;
}

template <class T> T &Max(const T &x, const T &y)
{
    return x < y ? y : x;
}

template <class T> T &Min(const T &x, const T &y)
{
    return x > y ? y : x;
}

template <class T> T Exp(const T &x)
{
    if (x == 0)
        return 1;
    const auto input = Abs(x);
    T result = 0;
    T numerator = 1;
    std::size_t denominator = 1;
    std::size_t i = 1;
    T term = numerator / denominator;
    while (i < 501 && term >= 1E-20)
    {
        result += term;
        if (denominator >= 1000)
        {
            numerator /= denominator;
            denominator = 1;
        }
        numerator *= input;
        denominator *= i;
        i++;
        term = numerator / denominator;
    }
    return x > 0 ? result : 1 / result;
}

template <class T> T NaturalLog(const T &x)
{
    if (x <= 0)
        throw ChiaRuntime::InvalidArgument("NaturalLog: Expected the input to be positive.");
    T input = x;
    T exp = 0;
    while (input > 1 && exp < 10000)
    {
        input /= 2;
        exp++;
    }
    input = input - 1;
    bool positiveTerm = true;
    T result = 0;
    T numerator = input;
    T denominator = 1;
    T ratio = numerator / denominator;
    for (std::size_t i = 0; i < 1000; i++)
    {
        result += ratio * (positiveTerm ? 1 : -1);
        numerator *= input;
        denominator++;
        ratio = numerator / denominator;
        positiveTerm = !positiveTerm;
    }
    return result + (Constants::Ln2)*exp;
}

template <class T> T Sine(const T &x)
{
    T input = x < 0 ? -x : x;
    const auto doublePI = Constants::Pi * 2;
    while (input >= doublePI)
        input -= doublePI;
    while (input <= -doublePI)
        input += doublePI;
    T squaredInput = input * input;
    T factor = 1;
    T numerator = input;
    T denominator = 1;
    T result = numerator / denominator;
    std::size_t i = 3;
    while (i < 2006)
    {
        factor = -factor;
        numerator *= squaredInput;
        denominator *= i * (i - 1);
        if (denominator > 10000)
        {
            numerator /= denominator;
            denominator = 1;
        }
        i += 2;
        result += factor * numerator / denominator;
    }
    return x < 0 ? -result : result;
}

template <class T> T Cosine(const T &x)
{
    T input = x < 0 ? -x : x;
    const auto doublePI = Constants::Pi * 2;
    while (input >= doublePI)
        input -= doublePI;
    while (input <= -doublePI)
        input += doublePI;
    T squaredInput = input * input;
    T factor = 1;
    T numerator = 1;
    T denominator = 1;
    T result = numerator / denominator;
    std::size_t i = 2;
    while (i < 2003)
    {
        factor = -factor;
        numerator *= squaredInput;
        denominator *= i * (i - 1);
        if (denominator > 10000)
        {
            numerator /= denominator;
            denominator = 1;
        }
        i += 2;
        result += factor * numerator / denominator;
    }
    return result;
}

template <class T> T Tangent(const T &x)
{
    return Sine(x) / Cosine(x);
}

template <class T> T Sinh(const T &x)
{
    const T exponential = Exp(x);
    return (exponential - 1 / exponential) * 0.5;
}

template <class T> T Cosh(const T &x)
{
    const T exponential = Exp(x);
    return (exponential + 1 / exponential) * 0.5;
}

template <class T> T Tanh(const T &x)
{
    if (2 * x > 14 || 2 * x < -14)
        return x > 0 ? 1 : -1;
    const T exponential = Exp(2 * x);
    return (exponential - 1) / (exponential + 1);
}

namespace
{
template <class T> T _PowerLong(const T &x, long n)
{
    if (x == 0 && n <= 0)
    {
        StringStream ss;
        ss << "Power: " << x;
        ss << " to the power of " << n;
        ss << " is undefined.";
        throw ChiaRuntime::InvalidArgument(ss.ToString());
    }
    else if (x == 0 || x == 1)
        return x;
    else if (n == 0)
        return 1;
    auto p = n > 0 ? n : -n;
    ChiaData::List<bool> even;
    while (p > 1)
    {
        even.Append((p & 1) == 0);
        p >>= 1;
    }
    auto result = x;
    for (auto itr = even.First(); itr != even.Last(); itr++)
    {
        result *= result;
        if (!(*itr))
            result *= x;
    }
    return n > 0 ? result : T(1) / result;
}
} // namespace

template <class T, class PowerType> T Power(const T x, PowerType n)
{
    if (x == 0 && n > 0)
        return 0;
    else if (x > 0)
    {
        if (n == 0)
            return 1;
        else if (n == 1)
            return x;
        else if (n < 0)
            return T(1) / Power(x, -n);
        return Exp(n * NaturalLog(x));
    }
    else if (x < 0 && (long)n == n)
        return _PowerLong<T>(x, (long)n);
    StringStream ss;
    ss << "Power: " << x;
    ss << " to the power of " << n;
    ss << " is undefined.";
    throw ChiaRuntime::InvalidArgument(ss.ToString());
}

template <class T> T ReLU(const T &x)
{
    return x < 0 ? 0 : x;
}

template <class T> T Sigmoid(const T &x)
{
    return 1 / (1 + Exp(-x));
}

template <class T> ChiaData::Math::Vector<T> Softmax(const ChiaData::Math::Vector<T> &vector)
{
    const T denomerator = Exp(vector).Sum();
    const ChiaData::Vector<T> numerator = Exp(vector);
    return numerator / denomerator;
}

template <class T> ChiaData::Math::Matrix<T> Softmax(const ChiaData::Math::Matrix<T> &matrix)
{
    const auto exponents = Exp(matrix);
    const auto summation = exponents.Sum();
    return exponents / summation;
}

template <class T> T Gauss(const T &x, const T &mu, const T &sigma)
{
    const T normalization = (x - mu) / sigma;
    return 1 / (sigma * Power(2 * Constants::Pi, 0.5)) * Exp(-0.5 * normalization * normalization);
}
} // namespace ChiaMath

#endif

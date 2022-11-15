#ifndef ML_ALGS_HPP
#define ML_ALGS_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <functional>

#include "DynamicArray.hpp"
#include "Math.hpp"
#include "Math/Matrix.hpp"
#include "Math/Vector.hpp"

using namespace ChiaData::Math;

namespace ChiaML
{
/*
Sign function.
@param value a value.
@return +1 if the value is positive, or 0 if the value is 0. -1, otherwise.
*/
template <class T> T Sign(T value);

/*
Generates a Vector that represents a given value using One Hot encoding.
@param value an unsigned integer to be encoded.
@param k the maximum value the value can take.
@return a Vector of K binary features that represents the given value.
*/
Vector<int> OneHot(size_t value, size_t k);

/**
 * Computes the Hinge Loss
 * @param prediction the prediction.
 * @param label the expected outcome.
 * @return the hinge loss given the prediction and label.
 **/
template <class T> T HingeLoss(const T &prediction, const T &label);

/**
 * Computes the gradient of Hinge Loss with respect to model weights.
 * @param input the input to the model.
 * @param prediction the prediction the model has made.
 * @param label the expected output.
 * @return the gradient of Hinge Loss with resect to model weights.
 **/
template <class InputType, class LabelType>
InputType HingeLossGradient(const InputType &input, const LabelType &prediction, const LabelType &label);

/*
The Perceptron Algorithm.
@param data a Matrix with n rows and d columns where n is the number of data
points and d is the dimensions of them.
@param labels a Vector with a single row and n columns indicating the labels of
the data points.
@param T the number of iterations to repeat the algorithm (default: 1000).
@return a Vector packed with the parameters (theta and offset at the end).
*/
template <class DataType, class LabelType>
Vector<double> Perceptron(const Matrix<DataType> &data, const Vector<LabelType> &labels, std::size_t T = 1000);

/*
The Averaged Perceptron Algorithm.
@param data a Matrix with n rows and d columns where n is the number of data
points and d is the dimensions of them.
@param labels a Vector with a single row and n columns indicating the labels of
the data points.
@param T the number of iterations to repeat the algorithm (default: 1000).
@return a Vector packed with the averaged parameters (theta and offset at the
end).
*/
template <class DataType, class LabelType>
Vector<double> AveragedPerceptron(const Matrix<DataType> &data, const Vector<LabelType> &labels, std::size_t T = 1000);

/*
Gradient Descent
@param f a function whose output will be minimized.
@param df a function that outputs the gradient of f.
@param initialX the initial input to f.
@param stepFunc a function that takes the current number of iterations and
returns the step size the gradient descent algorithm should take.
@param iterations the number of iterations to perform gradient descent.
@param recordHistory a bool indicates whether the input to and the output
of f are recorded in xHistory and outputHistory, respectively.
@param xHistory a pointer to the List used to record the inputs to f
during the process if recordHistory is set to true.
@param outputHistory a pointer to the List used to record the outputs of
f during the process if recordHistory is set to true.
@return the value of the input of f at the final step of gradient descent.
*/
template <class InputType, class OutputType, class StepType>
InputType GradientDescent(const std::function<OutputType(const InputType &)> &f,
                          const std::function<InputType(const InputType &)> &df, const InputType &initialX,
                          const std::function<StepType(std::size_t)> &stepFunc, std::size_t iterations,
                          bool recordHistory = false, ChiaData::DynamicArray<InputType> *xHistory = nullptr,
                          ChiaData::DynamicArray<OutputType> *outputHistory = nullptr);

/**
 * @brief Generate a positional encoding matrix
 *
 * @tparam FloatType the type of elements of the encoding matrix.
 * @param length the length of the sequence of words.
 * @param depth the depth (length) of the encoding.
 * @return Matrix<FloatType> the positional encoding matrix where the n-th row corresponds to the n-th word in the
 * sequence.
 */
template <class FloatType> Matrix<FloatType> PositionalEncoding(size_t length, size_t depth);
} // namespace ChiaML

#define MAX(a, b) (a) > (b) ? (a) : (b)

namespace ChiaML
{
template <class T> T Sign(T value)
{
    return value == 0 ? 0 : (value > 0 ? 1 : -1);
}

Vector<int> OneHot(size_t value, size_t k)
{
    Vector<int> encoding(k, 0);
    encoding[value - 1] = 1;
    return encoding;
}

template <class T> T HingeLoss(const T &prediction, const T &label)
{
    return MAX(0, 1 - prediction * label);
}

template <class InputType, class LabelType>
InputType HingeLossGradient(const InputType &input, const LabelType &prediction, const LabelType &label)
{
    return prediction * label < 1 ? -prediction * input : InputType({0});
}

template <class DataType, class LabelType>
Vector<double> Perceptron(const Matrix<DataType> &data, const Vector<LabelType> &labels, std::size_t T)
{
    const auto dataShape = data.Shape();
    const auto n = dataShape[0];
    const auto d = dataShape[1];
    Vector<double> th(d, 0.0);
    double th0 = 0.0;
    for (std::size_t i = 0; i < T; i++)
        for (std::size_t j = 0; j < n; j++)
            if (Sign(th.Dot(data[j]) + th0) != labels[j])
            {
                th += data[j] * labels[j];
                th0 += labels[j];
            }
    return Vector<double>::Combine({std::move(th), Vector<double>(1, th0)});
}

template <class DataType, class LabelType>
Vector<double> AveragedPerceptron(const Matrix<DataType> &data, const Vector<LabelType> &labels, std::size_t T)
{
    const auto dataShape = data.Shape();
    const auto n = dataShape[0];
    const auto d = dataShape[1];
    Vector<double> th(d, 0.0);
    Vector<double> ths(d, 0.0);
    double th0 = 0.0;
    double th0s = 0.0;
    for (std::size_t i = 0; i < T; i++)
        for (std::size_t j = 0; j < n; j++)
        {
            if (Sign(th.Dot(data[j]) + th0) != labels[j])
            {
                th += data[j] * labels[j];
                th0 += labels[j];
            }
            ths += th;
            th0s += th0;
        }
    const auto totalIterations = n * T;
    ths /= totalIterations;
    th0s /= totalIterations;
    return Vector<double>::Combine({std::move(ths), Vector<double>(1, th0s)});
}

template <class InputType, class OutputType, class StepType>
InputType GradientDescent(const std::function<OutputType(const InputType &)> &f,
                          const std::function<InputType(const InputType &)> &df, const InputType &initialX,
                          const std::function<StepType(std::size_t)> &stepFunc, std::size_t iterations,
                          bool recordHistory, ChiaData::DynamicArray<InputType> *xHistory,
                          ChiaData::DynamicArray<OutputType> *outputHistory)
{
    InputType x = initialX;
    if (recordHistory && xHistory)
        xHistory->Append(x);
    if (recordHistory && outputHistory)
        outputHistory->Append(f(x));
    for (std::size_t i = 1; i < iterations + 1; i++)
    {
        x -= df(x) * stepFunc(i);
        if (recordHistory && xHistory)
            xHistory->Append(x);
        if (recordHistory && outputHistory)
            outputHistory->Append(f(x));
    }
    return x;
}

template <class FloatType> Matrix<FloatType> PositionalEncoding(size_t length, size_t depth)
{
    Matrix<FloatType> result(length, depth);
    size_t position, i;
#pragma omp parallel for private(i) collapse(2) schedule(dynamic)
    for (position = 0; position < length; position++)
    {
        for (i = 0; i < (depth >> 1); i++)
        {
            const auto doubleI = i << 1;
            const FloatType value = position / ChiaMath::Power(FloatType(10000), FloatType(doubleI) / depth);
            result[position][doubleI] = ChiaMath::Sine(value);
            result[position][doubleI + 1] = ChiaMath::Cosine(value);
        }
    }
    return result;
}
} // namespace ChiaML

#endif

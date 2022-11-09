#include "NeuralNetwork/Layers/NLLLayer.hpp"
#include "Math.hpp"
#include "Math/Matrix.hpp"

#define LOG_EPSILON 1.0E-18

namespace ChiaML
{
namespace NN
{
namespace Layers
{
NLLLayer::NLLLayer()
{
}

double NLLLayer::ComputeLoss(const Matrix<double> &prediction,
                             const Matrix<double> &labels)
{
    return -prediction.Map([](const double &e) { return ChiaMath::NaturalLog(e + LOG_EPSILON); }).Scale(labels).SumAll();
}

Matrix<double> NLLLayer::Backward(const Matrix<double> &prediction,
                                  const Matrix<double> &labels)
{
    return prediction - labels;
}

std::string NLLLayer::ToString() const
{
    return "NLLLayer";
}
} // namespace Layers
} // namespace NN
} // namespace ChiaML

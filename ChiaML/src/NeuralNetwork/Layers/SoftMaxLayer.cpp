#include "NeuralNetwork/Layers/SoftMaxLayer.hpp"

#include "Math.hpp"

#include <sstream>

namespace ChiaML
{
namespace NN
{
namespace Layers
{
SoftMaxLayer::SoftMaxLayer()
{
}

Matrix<double> SoftMaxLayer::Forward(const Matrix<double> &input)
{
    this->input = input;
    const auto exponential = input.Map([](const double &e) { return ChiaMath::Exp(e); });
    return this->output = exponential / exponential.Sum();
}

Matrix<double> SoftMaxLayer::Backward(const Matrix<double> &derivative)
{
    return derivative;
}

std::string SoftMaxLayer::ToString() const
{
    std::stringstream ss;
    ss << "SoftMaxLayer:\n";
    ss << "Input\n" << this->input << std::endl;
    ss << "Output\n" << this->output;
    return ss.str();
}
} // namespace Layers
} // namespace NN
} // namespace ChiaML

#include "NeuralNetwork/Layers/TanhLayer.hpp"

#include <sstream>

#include "Math.hpp"

namespace ChiaML
{
namespace NN
{
namespace Layers
{
TanhLayer::TanhLayer()
{
}

Matrix<double> TanhLayer::Forward(const Matrix<double> &input)
{
    this->input = input;
    return this->output = input.Map([](const double &e) { return ChiaMath::Tanh(e); });
}

Matrix<double> TanhLayer::Backward(const Matrix<double> &derivative)
{
    return this->output.Map([](const double &e) { return 1 - e * e; }).Scale(derivative);
}

std::string TanhLayer::ToString() const
{
    std::stringstream ss;
    ss << "TanhLayer:" << std::endl;
    ss << "Input:\n" << this->input << std::endl;
    ss << "Output:\n" << this->output;
    return ss.str();
}
} // namespace Layers
} // namespace NN
} // namespace ChiaML

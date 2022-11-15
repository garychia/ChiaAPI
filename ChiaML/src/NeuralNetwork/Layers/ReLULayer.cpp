#include "NeuralNetwork/Layers/ReLULayer.hpp"

#include <sstream>

namespace ChiaML
{
namespace NN
{
namespace Layers
{
ReLULayer::ReLULayer()
{
}

Matrix<double> ReLULayer::Forward(const Matrix<double> &input)
{
    this->input = input;
    return this->output = input.Map([](const double &e) { return e > 0 ? e : 0; });
}

Matrix<double> ReLULayer::Backward(const Matrix<double> &derivative)
{
    return this->output.Map([](const double &e) { return e > 0 ? 1 : 0; }).Scale(derivative);
}

std::string ReLULayer::ToStdString() const
{
    std::stringstream ss;
    ss << "ReLULayer:\n";
    ss << "Input:\n" << this->input << std::endl;
    ss << "Ouput:\n" << this->output;
    return ss.str();
}
} // namespace Layers
} // namespace NN
} // namespace ChiaML

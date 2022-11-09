#include "NeuralNetwork/Layers/NeuralLayer.hpp"

namespace ChiaML
{
namespace NN
{
namespace Layers
{
NeuralLayer::NeuralLayer() : input(), output()
{
}

Matrix<double> NeuralLayer::GetLastInput() const
{
    return input;
}

Matrix<double> NeuralLayer::GetLastOutput() const
{
    return output;
}

std::ostream &operator<<(std::ostream &stream, const NeuralLayer &layer)
{
    stream << layer.ToString();
    return stream;
}
} // namespace Layers
} // namespace NN
} // namespace ChiaML

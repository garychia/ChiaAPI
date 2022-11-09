#include "NeuralNetwork/Layers/LinearLayer.hpp"

#include "Math.hpp"
#include "Random.hpp"

#include <sstream>

namespace ChiaML
{
namespace NN
{
namespace Layers
{
LinearLayer::LinearLayer() : NeuralLayer(), weights(), biases(), dWeights(), dBiases()
{
}

LinearLayer::LinearLayer(std::size_t inputSize, std::size_t outputSize)
    : NeuralLayer(), weights(inputSize, outputSize), biases(outputSize, 1), dWeights(), dBiases()
{
    // Using Xavier Weight Initialization.
    const auto sigma = ChiaMath::Power<double, double>(inputSize, -0.5);
    weights = weights.Map([&sigma](const double &) { return ChiaMath::Random::NormalDistribution(0, sigma); });
}

Matrix<double> LinearLayer::Forward(const Matrix<double> &input)
{
    // Record the input.
    this->input = input;
    // Calculate the output.
    return this->output = weights.Transposed() * input + biases;
}

Matrix<double> LinearLayer::Backward(const Matrix<double> &derivative)
{
    // Calculate the derivatives.
    dWeights = this->input * derivative.Transposed();
    dBiases = derivative.Sum(false);
    // Calculate the derivatives with respect to the output of previous layer.
    return this->weights * derivative;
}

void LinearLayer::UpdateWeights(const double &learningRate)
{
    // Update the weights and biases.
    weights -= learningRate * dWeights;
    biases -= learningRate * dBiases;
}

std::string LinearLayer::ToString() const
{
    std::stringstream ss;
    ss << "LinearLayer: {" << std::endl;
    ss << "  Weights: {\n";
    ss << "    ";
    for (const auto &c : weights.ToStdString())
    {
        ss << c;
        if (c == '\n')
            ss << "    ";
    }
    ss << "\n  },\n";
    ss << "  Biases: {\n";
    ss << "    ";
    for (const auto &c : biases.ToStdString())
    {
        ss << c;
        if (c == '\n')
            ss << "    ";
    }
    ss << "\n  }\n";
    ss << "}";
    return ss.str();
}
} // namespace Layers
} // namespace NN
} // namespace ChiaML

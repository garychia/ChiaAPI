#ifndef LINEARLAYER_HPP
#define LINEARLAYER_HPP

#include "NeuralLayer.hpp"

using namespace ChiaData;

namespace ChiaML
{
namespace NN
{
namespace Layers
{
/**
 * Linear Neural Network Layer.
 **/
class LinearLayer : public NeuralLayer
{
  private:
    // Matrix whose columns represent a unit.
    Matrix<double> weights;
    // Column vector whose elements are the biases of the units.
    Matrix<double> biases;
    // Derivative with respect to the weights.
    Matrix<double> dWeights;
    // Derivative with respect to the biases.
    Matrix<double> dBiases;

  public:
    // LinearLayer Contructor.
    LinearLayer();
    /**
     * LinearLayer Contructor
     * @param inputSize the dimension of the input.
     * @param outputSize the dimension of the output of this layer.
     **/
    LinearLayer(std::size_t inputSize, std::size_t outputSize);
    // LinearLayer Destructor.
    virtual ~LinearLayer() = default;
    /**
     * Generate an output based on the weights and biases.
     * @param input the input to this layer.
     * @return the output of this layer.
     **/
    Matrix<double> Forward(const Matrix<double> &input) override;
    /**
     * Backpropogate the loss and compute the derivative of the weights.
     * @param derivative the derivative of the next layer.
     * @return the derivative with respect to the output of the previous layer.
     **/
    virtual Matrix<double> Backward(const Matrix<double> &derivative) override;
    /**
     * Update the weights of this layer using gradient descent.
     * @param learningRate the learning rate of gradient descent.
     **/
    virtual void UpdateWeights(const double &learningRate) override;
    /**
     * Generate a string description of this layer.
     * @return a string that describes this layer.
     **/
    std::string ToString() const override;
};
} // namespace Layers
} // namespace NN
} // namespace ChiaML

#endif // NEURALLAYER_HPP

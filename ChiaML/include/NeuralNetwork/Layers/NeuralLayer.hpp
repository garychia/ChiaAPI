#ifndef NEURALLAYER_HPP
#define NEURALLAYER_HPP

#include "Math/Matrix.hpp"

#include <ostream>

using namespace ChiaData::Math;

namespace ChiaML
{
namespace NN
{
namespace Layers
{
/**
 * (Dense) Neural Network Layer.
 **/
class NeuralLayer
{
  protected:
    // Input to this layer.
    Matrix<double> input;
    // Output of this layer.
    Matrix<double> output;

  public:
    // NeuralLayer Contructor.
    NeuralLayer();
    virtual ~NeuralLayer() = default;
    /**
     * Generate an output based on the weights and biases.
     * @param input the input to this layer.
     * @return the output of this layer.
     **/
    virtual Matrix<double> Forward(const Matrix<double> &input) = 0;
    /**
     * Backpropogate the loss and compute the derivative of the weights.
     * @param derivative the derivative of the next layer.
     * @return the derivative with respect to the output of the previous layer.
     **/
    virtual Matrix<double> Backward(const Matrix<double> &derivative) = 0;
    /**
     * Update the weights of this layer using gradient descent.
     * @param learningRate the learning rate of gradient descent.
     **/
    virtual void UpdateWeights(const double &learningRate) = 0;
    /**
     * Generate a string description of this layer.
     * @return a string that describes this layer.
     **/
    virtual std::string ToString() const = 0;

    /**
     * Retrieve the lastest input to this layer.
     * @return the lastest input to this layer. If this layer has
     * not received any input yet, an empty matrix is returned.
     **/
    Matrix<double> GetLastInput() const;

    /**
     * Retrieve the lastest output of this layer.
     * @return the lastest output of this layer. If this layer has
     * not produced any output yet, an empty matrix is returned.
     **/
    Matrix<double> GetLastOutput() const;

    friend std::ostream &operator<<(std::ostream &stream, const NeuralLayer &layer);
};
} // namespace Layers
} // namespace NN
} // namespace ChiaML

#endif // NEURALLAYER_HPP

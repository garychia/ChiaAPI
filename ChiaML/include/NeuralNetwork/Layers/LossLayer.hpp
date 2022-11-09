#ifndef LOSSLAYER_HPP
#define LOSSLAYER_HPP

#include "NeuralLayer.hpp"

#include <string>

namespace ChiaML
{
namespace NN
{
namespace Layers
{
class LossLayer
{
  public:
    // LossLayer Constructor
    LossLayer();
    // LossLayer Destructor
    virtual ~LossLayer() = default;
    /**
     * Compute the loss based on the prediction and labels.
     * @param prediction the prediction a neural network has generated.
     * @param labels the correct output the network should generate.
     * @return the loss.
     **/
    virtual double ComputeLoss(const Matrix<double> &prediction, const Matrix<double> &labels) = 0;

    /**
     * Compute the derivative with respect to the prediction a neural network has generated.
     * @param prediction the prediction from the network.
     * @param labels the correct output the network should generate.
     * @return the derivative.
     **/
    virtual Matrix<double> Backward(const Matrix<double> &prediction,
                                              const Matrix<double> &labels) = 0;

    /**
     * Generate a string that describes this layer.
     * @return a string that describes this layer.
     **/
    virtual std::string ToString() const = 0;
};
} // namespace Layers
} // namespace NN
} // namespace ChiaML

#endif // LOSSLAYER_HPP

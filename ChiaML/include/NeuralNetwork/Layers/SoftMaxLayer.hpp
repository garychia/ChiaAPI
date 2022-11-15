#ifndef SOFTMAXLAYER_HPP
#define SOFTMAXLAYER_HPP

#include "ActivationLayer.hpp"

namespace ChiaML
{
namespace NN
{
namespace Layers
{
class SoftMaxLayer : public ActivationLayer
{
  public:
    SoftMaxLayer();
    ~SoftMaxLayer() = default;
    /**
     * Compute the ReLU value of each element of the input matrix.
     * @param input the input to this layer.
     * @return the output of this layer.
     **/
    virtual Matrix<double> Forward(const Matrix<double> &input) override;
    /**
     * Backpropogate the loss.
     * @param derivative the derivative of the next layer.
     * @return the derivative with respect to the output of the previous layer.
     **/
    virtual Matrix<double> Backward(const Matrix<double> &derivative) override;
    /**
     * Generate a string description of this layer.
     * @return a string that describes this layer.
     **/
    virtual std::string ToStdString() const override;
};
} // namespace Layers
} // namespace NN
} // namespace ChiaML

#endif // SOFTMAXLAYER_HPP

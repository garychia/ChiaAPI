#ifndef TANH_HPP
#define TANH_HPP

#include "ActivationLayer.hpp"

namespace ChiaML
{
namespace NN
{
namespace Layers
{
class TanhLayer : public ActivationLayer
{
  private:
    /* data */
  public:
    // TanhLayer Constructor
    TanhLayer();
    ~TanhLayer() = default;
    /**
     * Compute the Tanh value of each element of the input matrix.
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
    virtual std::string ToString() const override;
};
} // namespace Layers
} // namespace NN
} // namespace ChiaML

#endif // TANH_HPP

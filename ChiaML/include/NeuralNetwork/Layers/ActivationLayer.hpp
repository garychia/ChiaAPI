#ifndef ACTIVATIONLAYER_HPP
#define ACTIVATIONLAYER_HPP

#include "NeuralLayer.hpp"

namespace ChiaML
{
namespace NN
{
namespace Layers
{
class ActivationLayer : public NeuralLayer
{
  public:
    ActivationLayer();
    virtual ~ActivationLayer() = default;
    virtual void UpdateWeights(const double &learningRate) override;
};
} // namespace Layers
} // namespace NN
} // namespace ChiaML

#endif

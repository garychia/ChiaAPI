#include "NeuralLayer.hpp"
#include "Random.hpp"

namespace ChiaML
{
namespace NN
{
namespace Layers
{
class BatchNormLayer : public NeuralLayer
{
  private:
    // Mean of the input.
    Matrix<double> mean;
    // Variance of the input.
    Matrix<double> variance;
    // Normalized Input.
    Matrix<double> normalized;
    // Scaling factor for the normalized input.
    Matrix<double> scale;
    // Shifting factor for the normalized input.
    Matrix<double> shift;
    // Derivative with respect to the scaling factor.
    Matrix<double> dScale;
    // Derivative with respect to the shifting factor.
    Matrix<double> dShift;

  public:
    /**
     * Constructor
     * @param inputSize the size of the input.
     **/
    BatchNormLayer(std::size_t inputSize);

    ~BatchNormLayer() = default;

    /**
     * Compute the batch-normalized output given some input.
     * @param input the input to this layer.
     * @return the output of this layer.
     **/
    virtual Matrix<double> Forward(const Matrix<double> &input) override;

    /**
     * Backpropogate the loss and compute the derivative of the scaling and shifting factors.
     * @param derivative the derivative of the next layer.
     * @return the derivative with respect to the output of the previous layer.
     **/
    virtual Matrix<double> Backward(const Matrix<double> &derivative) override;

    /**
     * Update the scaling and shifting factors of this layer using gradient descent.
     * @param learningRate the learning rate of gradient descent.
     **/
    virtual void UpdateWeights(const double &learningRate) override;

    /**
     * Generate a string description of this layer.
     * @return a string that describes this layer.
     **/
    virtual std::string ToStdString() const override;
};
} // namespace Layers
} // namespace NN
} // namespace ChiaML

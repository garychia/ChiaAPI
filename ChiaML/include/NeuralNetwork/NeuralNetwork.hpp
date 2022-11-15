#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "DynamicArray.hpp"
#include "Layers/LossLayer.hpp"
#include "Layers/NeuralLayer.hpp"

#include <sstream>
#include <string>

namespace ChiaML
{
namespace NN
{
// Neural Network
class NeuralNetwork
{
  private:
    // Layers
    ChiaData::DynamicArray<ChiaML::NN::Layers::NeuralLayer *> layers;
    // Loss function
    ChiaML::NN::Layers::LossLayer *lossLayer;

    void Learn(const Matrix<double> &derivative, const double &learningRate);

  public:
    enum class LayerType
    {
        Linear,
        ReLU,
        SoftMax,
        Tanh,
        BatchNormalization
    };

    enum class LossType
    {
        NLL // Negative Log-Likelihood
    };

    /* NeuralNetwork Constructor */
    NeuralNetwork();

    /**
     * NeuralNetwork Constructor
     * @param layerTypes a list of LayerType to specify the type of each layer.
     * @param shapes a list of Tuple to specify the input and output size of each layer.
     * @param lossType the loss function to be used to evaluate loss.
     **/
    NeuralNetwork(const ChiaData::DynamicArray<LayerType> &layerTypes,
                  const ChiaData::DynamicArray<Tuple<unsigned int>> &shapes, LossType lossType);

    // NeuralNetwork Destructor
    ~NeuralNetwork();

    /**
     * Make a prediction based on the input and the layers.
     * @param input the input to this network.
     * @return a Matrix as the output.
     **/
    Matrix<double> Predict(const Matrix<double> &input);

    /**
     * Train this NeuralNetwork.
     * @param trainingData the training data used to train this network.
     * @param labels the correct output the network should produce.
     * @param epochs the number of epochs (iterations through the training data).
     * @param learningRate the learning rate of gradient descent.
     * @return a list of loss values recorded during the training.
     **/
    ChiaData::DynamicArray<double> Train(const Matrix<double> &trainingData, const Matrix<double> &labels,
                                         unsigned int epochs, double learningRate = 0.001);

    /**
     * Generate a string that describes this NeuralNetwork.
     * @return a string that describes this NeuralNetwork.
     **/
    std::string ToStdString() const;

    friend std::ostream &operator<<(std::ostream &stream, const NeuralNetwork &network);
};

std::ostream &operator<<(std::ostream &stream, const NeuralNetwork::LayerType &layerType);
std::ostream &operator<<(std::ostream &stream, const NeuralNetwork::LossType &lossType);
} // namespace NN
} // namespace ChiaML

#endif // NEURALNETWORK_HPP

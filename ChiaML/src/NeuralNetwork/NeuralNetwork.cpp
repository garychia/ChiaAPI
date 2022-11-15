#include "NeuralNetwork/NeuralNetwork.hpp"
#include "Exceptions.hpp"
#include "NeuralNetwork/Layers/BatchNormLayer.hpp"
#include "NeuralNetwork/Layers/LinearLayer.hpp"
#include "NeuralNetwork/Layers/NLLLayer.hpp"
#include "NeuralNetwork/Layers/ReLULayer.hpp"
#include "NeuralNetwork/Layers/SoftMaxLayer.hpp"
#include "NeuralNetwork/Layers/TanhLayer.hpp"
#include "Random.hpp"

#include <ostream>

using namespace ChiaML::NN::Layers;

namespace ChiaML
{
namespace NN
{
NeuralNetwork::NeuralNetwork() : layers(), lossLayer(nullptr)
{
}

NeuralNetwork::NeuralNetwork(const ChiaData::DynamicArray<LayerType> &layerTypes,
                             const ChiaData::DynamicArray<Tuple<unsigned int>> &shapes, LossType lossType)
    : layers(), lossLayer(nullptr)
{
    if (layerTypes.Length() != shapes.Length())
        throw ChiaRuntime::InvalidArgument("NeuralNetwork: Input and output size "
                                           "must be specified for each layer.");
    for (std::size_t i = 0; i < layerTypes.Length(); i++)
    {
        switch (layerTypes[i])
        {
        case LayerType::Linear: {
            if (shapes[i].Size() != 2)
                throw ChiaRuntime::InvalidArgument("NeuralNetwork: the input and output sizes of a LinearLayer must "
                                                   "be specified.");
            layers.Append(new LinearLayer(shapes[i][0], shapes[i][1]));
            break;
        }
        case LayerType::ReLU: {
            layers.Append(new ReLULayer());
            break;
        }
        case LayerType::SoftMax: {
            layers.Append(new SoftMaxLayer());
            break;
        }
        case LayerType::Tanh: {
            layers.Append(new TanhLayer());
            break;
        }
        case LayerType::BatchNormalization: {
            if (shapes[i].Size() != 1)
                throw ChiaRuntime::InvalidArgument("NeuralNetwork: the size of input to a BatchNormLayer must be "
                                                   "specified.");
            layers.Append(new BatchNormLayer(shapes[i][0]));
            break;
        }
        default: {
            throw ChiaRuntime::InvalidArgument("NeuralNetwork: Unexpected LayerType found.");
        }
        }
    }

    switch (lossType)
    {
    case LossType::NLL:
        lossLayer = new NLLLayer();
        break;
    default: {
        throw ChiaRuntime::InvalidArgument("NeuralNetwork: Unexpected LossType found.");
    }
    }
}

NeuralNetwork::~NeuralNetwork()
{
    for (std::size_t i = 0; i < layers.Length(); i++)
        delete layers[i];
    layers.RemoveAll();
    delete lossLayer;
}

Matrix<double> NeuralNetwork::Predict(const Matrix<double> &input)
{
    Matrix<double> output = input;
    for (std::size_t i = 0; i < layers.Length(); i++)
        output = layers[i]->Forward(output);
    return output;
}

void NeuralNetwork::Learn(const Matrix<double> &derivative, const double &learningRate)
{
    if (layers.IsEmpty())
        return;
    Matrix<double> currentDerivative = derivative;
    size_t i = layers.Length() - 1;
    while (true)
    {
        currentDerivative = layers[i]->Backward(currentDerivative);
        layers[i]->UpdateWeights(learningRate);
        if (i == 0)
            break;
        i--;
    }
}

ChiaData::DynamicArray<double> NeuralNetwork::Train(const Matrix<double> &trainingData, const Matrix<double> &labels,
                                                    unsigned int epochs, double learningRate)
{
    ChiaData::DynamicArray<double> lossHistory;
    for (std::size_t i = 0; i < epochs; i++)
    {
        const auto pred = Predict(trainingData);
        lossHistory.Append(lossLayer->ComputeLoss(pred, labels));
        const auto derivative = lossLayer->Backward(pred, labels);
        Learn(derivative, learningRate);
    }
    return lossHistory;
}

std::string NeuralNetwork::ToStdString() const
{
    std::stringstream ss;
    const auto nLayers = layers.Length();
    ss << "NeuralNetwork: {\n";
    ss << "  Number of Layers: " << nLayers << ",";
    if (nLayers > 0)
    {
        ss << std::endl << "  Layers: {\n";
        for (std::size_t i = 0; i < nLayers; i++)
        {
            ss << "    Layer " << i + 1 << ": {\n";
            ss << "      ";
            for (const auto &c : layers[i]->ToStdString())
            {
                ss << c;
                if (c == '\n')
                    ss << "      ";
            }
            ss << "\n    }";
            if (i < nLayers - 1)
                ss << ",";
            ss << "\n";
        }
        ss << "  },\n";
    }
    ss << "  Loss Layer: " << lossLayer->ToString() << std::endl;
    ss << "}";
    return ss.str();
}

std::ostream &operator<<(std::ostream &stream, const NeuralNetwork &network)
{
    stream << network.ToStdString();
    return stream;
}

std::ostream &operator<<(std::ostream &stream, const NeuralNetwork::LayerType &layerType)
{
    stream << static_cast<int>(layerType);
    return stream;
}

std::ostream &operator<<(std::ostream &stream, const NeuralNetwork::LossType &lossType)
{
    stream << static_cast<int>(lossType);
    return stream;
}
} // namespace NN
} // namespace ChiaML

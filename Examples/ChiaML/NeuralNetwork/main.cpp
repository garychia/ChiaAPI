#include <iostream>

#include "NeuralNetwork/NeuralNetwork.hpp"

#include <stdlib.h>

using namespace ChiaML::NN;
using namespace ChiaML;
using namespace ChiaData;

int main(void)
{
    Matrix<double> input({{2, 3, 9, 12}, {5, 2, 6, 5}});
    Matrix<double> labels({{0, 1, 0, 1}, {1, 0, 1, 0}});
    Array<NeuralNetwork::LayerType> layerTypes({NeuralNetwork::LayerType::Linear,
                                                NeuralNetwork::LayerType::BatchNormalization,
                                                NeuralNetwork::LayerType::SoftMax});
    Array<Tuple<unsigned int>> shapes({{2, 2}, {2}, {}});
    NeuralNetwork network(layerTypes, shapes, NeuralNetwork::LossType::NLL);
    std::cout << "Labels:" << std::endl;
    std::cout << labels << std::endl;
    std::cout << "Prediction Before Training:" << std::endl;
    std::cout << network.Predict(input) << std::endl;
    const auto epochs = 500;
    const double learningRate = 0.01;
    const auto history = network.Train(input, labels, epochs, learningRate);
    std::cout << "Loss History (First and Last):\n";
    std::cout << history[0] << std::endl;
    std::cout << history[history.Length() - 1] << std::endl;
    std::cout << "Prediction After Training:" << std::endl;
    std::cout << network.Predict(input) << std::endl;
    return 0;
}

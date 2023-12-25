
#include "neural_network.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(int numInputs, int numHidden, int numOutputs)
    : generator(std::random_device{}()), distribution(-1.0, 1.0)
{
    weightsInputHidden.resize(numInputs, std::vector<double>(numHidden));
    weightsHiddenOutput.resize(numHidden, std::vector<double>(numOutputs));
    biasHidden.resize(numHidden);
    biasOutput.resize(numOutputs);

    initializeWeightsAndBiases();
}

void NeuralNetwork::initializeWeightsAndBiases() {
    for (size_t i = 0; i < weightsInputHidden.size(); ++i) {
        for (size_t j = 0; j < weightsInputHidden[0].size(); ++j) {
            weightsInputHidden[i][j] = distribution(generator);
        }
    }
    for (size_t i = 0; i < weightsHiddenOutput.size(); ++i) {
        for (size_t j = 0; j < weightsHiddenOutput[0].size(); ++j) {
            weightsHiddenOutput[i][j] = distribution(generator);
        }
    }

    for (size_t i = 0; i < biasHidden.size(); ++i) {
        biasHidden[i] = distribution(generator);
    }
    for (size_t i = 0; i < biasOutput.size(); ++i) {
        biasOutput[i] = distribution(generator);
    }
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::dSigmoid(double y) {
    return y * (1.0 - y);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> hiddenOutputs(biasHidden.size());
    for (size_t i = 0; i < biasHidden.size(); ++i) {
        double sum = biasHidden[i];
        for (size_t j = 0; j < input.size(); ++j) {
            sum += input[j] * weightsInputHidden[j][i];
        }
        hiddenOutputs[i] = sigmoid(sum);
    }

    std::vector<double> finalOutputs(biasOutput.size());
    for (size_t i = 0; i < biasOutput.size(); ++i) {
        double sum = biasOutput[i];
        for (size_t j = 0; j < hiddenOutputs.size(); ++j) {
            sum += hiddenOutputs[j] * weightsHiddenOutput[j][i];
        }
        finalOutputs[i] = sigmoid(sum);
    }
    return finalOutputs;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            const std::vector<double>& input = inputs[i];
            const std::vector<double>& target = targets[i];

            // Feedforward
            std::vector<double> hiddenOutputs(biasHidden.size());
            for (size_t h = 0; h < biasHidden.size(); ++h) {
                double sum = biasHidden[h];
                for (size_t j = 0; j < input.size(); ++j) {
                    sum += input[j] * weightsInputHidden[j][h];
                }
                hiddenOutputs[h] = sigmoid(sum);
            }

            std::vector<double> finalOutputs(biasOutput.size());
            for (size_t o = 0; o < biasOutput.size(); ++o) {
                double sum = biasOutput[o];
                for (size_t h = 0; h < hiddenOutputs.size(); ++h) {
                    sum += hiddenOutputs[h] * weightsHiddenOutput[h][o];
                }
                finalOutputs[o] = sigmoid(sum);
            }

            // Backpropagation
            std::vector<double> outputErrors(biasOutput.size());
            for (size_t o = 0; o < biasOutput.size(); ++o) {
                outputErrors[o] = target[o] - finalOutputs[o];
            }

            std::vector<double> hiddenErrors(biasHidden.size());
            for (size_t h = 0; h < biasHidden.size(); ++h) {
                double sum = 0;
                for (size_t o = 0; o < biasOutput.size(); ++o) {
                    sum += outputErrors[o] * weightsHiddenOutput[h][o];
                }
                hiddenErrors[h] = sum;
            }

            for (size_t o = 0; o < biasOutput.size(); ++o) {
                biasOutput[o] += outputErrors[o] * dSigmoid(finalOutputs[o]) * learningRate;
                for (size_t h = 0; h < hiddenOutputs.size(); ++h) {
                    weightsHiddenOutput[h][o] += hiddenOutputs[h] * outputErrors[o] * dSigmoid(finalOutputs[o]) * learningRate;
                }
            }

            for (size_t h = 0; h < biasHidden.size(); ++h) {
                biasHidden[h] += hiddenErrors[h] * dSigmoid(hiddenOutputs[h]) * learningRate;
                for (size_t j = 0; j < input.size(); ++j) {
                    weightsInputHidden[j][h] += input[j] * hiddenErrors[h] * dSigmoid(hiddenOutputs[h]) * learningRate;
                }
            }
        }
    }
}

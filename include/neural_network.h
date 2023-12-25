
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <cmath>
#include <random>
#include <numeric>

class NeuralNetwork {
public:
    NeuralNetwork(int numInputs, int numHidden, int numOutputs);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs);
    std::vector<double> predict(const std::vector<double>& input);

private:
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasHidden;
    std::vector<double> biasOutput;
    double learningRate = 0.1;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;

    double sigmoid(double x);
    double dSigmoid(double y);
    void initializeWeightsAndBiases();
};

#endif // NEURAL_NETWORK_H

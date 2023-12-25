
#include <iostream>
#include "neural_network.h"
#include "astar_search.h"

int main() {
    std::cout << "Running C++ AI Algorithms Demo..." << std::endl;

    // Neural Network Demo
    std::cout << "
--- Neural Network Demo ---" << std::endl;
    NeuralNetwork nn(2, 3, 1);
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    nn.train(inputs, targets, 10000);

    std::cout << "Prediction for [0, 0]: " << nn.predict({0, 0})[0] << std::endl;
    std::cout << "Prediction for [0, 1]: " << nn.predict({0, 1})[0] << std::endl;
    std::cout << "Prediction for [1, 0]: " << nn.predict({1, 0})[0] << std::endl;
    std::cout << "Prediction for [1, 1]: " << nn.predict({1, 1})[0] << std::endl;

    // A* Search Demo (conceptual)
    std::cout << "
--- A* Search Demo (Conceptual) ---" << std::endl;
    // In a real implementation, this would involve a Graph, Nodes, and Edge weights.
    // AStarSearch astar(startNode, goalNode);
    // std::vector<Node> path = astar.findPath();
    std::cout << "A* Search would find the shortest path in a graph." << std::endl;

    return 0;
}


#ifndef ASTAR_SEARCH_H
#define ASTAR_SEARCH_H

#include <vector>
#include <string>
#include <map>
#include <queue>

// Forward declaration
class Node;

class Edge {
public:
    Node* target;
    double weight;

    Edge(Node* target, double weight) : target(target), weight(weight) {}
};

class Node {
public:
    std::string name;
    std::vector<Edge> neighbors;
    double g_score; // Cost from start to current node
    double h_score; // Heuristic cost from current node to goal
    double f_score; // g_score + h_score
    Node* parent;

    Node(std::string name) : name(name), g_score(INFINITY), h_score(INFINITY), f_score(INFINITY), parent(nullptr) {}

    void addNeighbor(Node* target, double weight) {
        neighbors.emplace_back(target, weight);
    }
};

// Comparator for priority queue
struct CompareNode {
    bool operator()(Node* a, Node* b) {
        return a->f_score > b->f_score;
    }
};

class AStarSearch {
public:
    AStarSearch(Node* start, Node* goal) : startNode(start), goalNode(goal) {}
    std::vector<Node*> findPath();

private:
    Node* startNode;
    Node* goalNode;
    // Heuristic function (e.g., Euclidean distance for grid-based search)
    double heuristic(Node* a, Node* b) {
        // For simplicity, a dummy heuristic
        return 0.0;
    }
};

#endif // ASTAR_SEARCH_H

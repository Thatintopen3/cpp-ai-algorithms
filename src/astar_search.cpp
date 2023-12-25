
#include "astar_search.h"
#include <algorithm>
#include <limits>

#define INFINITY std::numeric_limits<double>::infinity()

std::vector<Node*> AStarSearch::findPath() {
    std::priority_queue<Node*, std::vector<Node*>, CompareNode> openSet;
    std::map<Node*, Node*> cameFrom;
    std::map<Node*, double> gScore;

    startNode->g_score = 0;
    startNode->h_score = heuristic(startNode, goalNode);
    startNode->f_score = startNode->h_score;
    openSet.push(startNode);
    gScore[startNode] = 0;

    while (!openSet.empty()) {
        Node* current = openSet.top();
        openSet.pop();

        if (current == goalNode) {
            std::vector<Node*> path;
            while (current != nullptr) {
                path.push_back(current);
                current = cameFrom[current];
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (const Edge& edge : current->neighbors) {
            Node* neighbor = edge.target;
            double tentative_gScore = gScore[current] + edge.weight;

            if (tentative_gScore < gScore[neighbor]) {
                cameFrom[neighbor] = current;
                gScore[neighbor] = tentative_gScore;
                neighbor->g_score = tentative_gScore;
                neighbor->h_score = heuristic(neighbor, goalNode);
                neighbor->f_score = neighbor->g_score + neighbor->h_score;
                openSet.push(neighbor);
            }
        }
    }

    return {}; // No path found
}

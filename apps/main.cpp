#include "ml/layers/DenseLayer.h"
#include "ml/activations/ReLUActivation.h"
#include "ml/core/Vector.h"

#include <iostream>
#include <vector>

using namespace machineLearning;

void printVector(const std::string& name, const Vector& v) {
    std::cout << name << " (size " << v.size() << "): ";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    // ---- Input (size = 2) ----
    Vector input({1.0, -2.0});

    // ---- Layers ----
    DenseLayer d1(2, 4);     // (2 → 4)
    ReLUActivation relu(4);  // activation for size 4
    DenseLayer d2(4, 1);     // (4 → 1)

    // ---- Forward pass ----
    Vector z1 = d1.forward(input);
    Vector a1 = relu.forward(z1);
    Vector z2 = d2.forward(a1);

    // ---- Print results ----
    printVector("Input", input);
    printVector("Z1 (Dense1 output)", z1);
    printVector("A1 (ReLU output)", a1);
    printVector("Z2 (Final output)", z2);

    return 0;
}
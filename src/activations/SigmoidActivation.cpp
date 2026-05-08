#include "ml/activations/SigmoidActivation.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace machineLearning {

SigmoidActivation::SigmoidActivation(size_t inputSize)
    : inputCache(std::vector<double>(inputSize, 0.0))
{}

Vector SigmoidActivation::forward(const Vector& input) {
    if (input.size() != inputCache.size()) {
        throw std::invalid_argument("Input size does not match SigmoidActivation size");
    }

    inputCache = input;

    std::vector<double> outputData;
    outputData.reserve(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        double z = input[i];
        double value = 1.0 / (1.0 + std::exp(-z));
        outputData.push_back(value);
    }

    return Vector(std::move(outputData));
}

Vector SigmoidActivation::backward(const Vector& incomingGradient) {
    if (incomingGradient.size() != inputCache.size()) {
        throw std::invalid_argument("Incoming gradient size does not match SigmoidActivation size");
    }

    std::vector<double> outputGradient;
    outputGradient.reserve(incomingGradient.size());

    for (size_t i = 0; i < incomingGradient.size(); ++i) {
        double z = inputCache[i];
        double sigmoid = 1.0 / (1.0 + std::exp(-z));
        double derivative = sigmoid * (1.0 - sigmoid);
        outputGradient.push_back(incomingGradient[i] * derivative);
    }

    return Vector(std::move(outputGradient));
}

}
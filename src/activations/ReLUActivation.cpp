#include "ml/activations/ReLUActivation.h"

#include <vector>
#include <stdexcept>
#include <utility>
#include <algorithm>

namespace machineLearning {

ReLUActivation::ReLUActivation(size_t inputSize) 
    : inputCache(std::vector<double>(inputSize, 0.0))
{
}

Vector ReLUActivation::forward(const Vector& input) {
    
    if (input.size() != inputCache.size()) {
        throw std::invalid_argument("Input size does not match ReLUActivation size");
    }
    
    inputCache = input;

    const size_t inputSize = input.size();
    std::vector<double> maxRes(inputSize);
    
    for (size_t i = 0; i < inputSize; ++i) {
        maxRes[i] = input[i] > 0.0 ? input[i] : 0.01 * input[i];
    }
    return Vector(std::move(maxRes));
}

Vector ReLUActivation::backward(const Vector& incomingGradient) {
    if (incomingGradient.size() != inputCache.size()) {
        throw std::invalid_argument("Incoming gradient size does not match ReLUActivation size");
    }

    std::vector<double> outputGradientData(incomingGradient.size());

    for (size_t i = 0; i < incomingGradient.size(); ++i) {
        outputGradientData[i] = inputCache[i] > 0.0
    ? incomingGradient[i]
    : 0.01 * incomingGradient[i];
    }

    return Vector(std::move(outputGradientData));
}

}
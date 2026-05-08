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
        maxRes[i] = std::max(0.0, input[i]);
    }
    return Vector(std::move(maxRes));
}

}
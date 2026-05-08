#include "ml/layers/DenseLayer.h"
#include "ml/core/Operations.h"

#include <vector>
#include <random>
#include <stdexcept>

namespace machineLearning {

namespace {

std::vector<double> generateWeights(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.01, 0.01);

    std::vector<double> data;
    data.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        data.push_back(dist(gen));
    }

    return data;
}

} // namespace

DenseLayer::DenseLayer(size_t inputSize, size_t outputSize)
    : weights(outputSize,
              inputSize,
              generateWeights(outputSize * inputSize)),
      bias(std::vector<double>(outputSize, 0.0)),
      inputCache(std::vector<double>(inputSize, 0.0))
{
}

const Matrix& DenseLayer::getWeights() const {
    return weights;
}

const Vector& DenseLayer::getBias() const {
    return bias;
}

Vector DenseLayer::forward(const Vector& input) {
    if (input.size() != weights.colCount()) {
        throw std::invalid_argument("Input size does not match DenseLayer input size");
    }
    
    inputCache = input;
    Vector output = weights * input;

    for (size_t i = 0; i < bias.size(); ++i) {
        output[i] += bias[i];
    }

    return output;
}

}
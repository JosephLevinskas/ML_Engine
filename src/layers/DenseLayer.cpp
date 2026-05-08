#include "ml/layers/DenseLayer.h"
#include "ml/core/Operations.h"

#include <vector>
#include <random>
#include <stdexcept>
#include <utility>

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

} 

DenseLayer::DenseLayer(size_t inputSize, size_t outputSize)
    : weights(outputSize,
              inputSize,
              generateWeights(outputSize * inputSize)),
      bias(std::vector<double>(outputSize, 0.0)),
      weightGradients(outputSize,
                      inputSize,
                      std::vector<double>(outputSize * inputSize, 0.0)),
      biasGradients(std::vector<double>(outputSize, 0.0)),
      inputCache(std::vector<double>(inputSize, 0.0))
{}

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

Vector DenseLayer::backward(const Vector& incomingGradient) {
    if (incomingGradient.size() != bias.size()) {
        throw std::invalid_argument("Incoming gradient size does not match DenseLayer output size");
    }

    const size_t outputSize = weights.rowCount();
    const size_t inputSize = weights.colCount();

    // dL/dW[i][j] = dL/dz[i] * input[j]
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weightGradients(i, j) = incomingGradient[i] * inputCache[j];
        }
    }

    // dL/db = dL/dz
    for (size_t i = 0; i < outputSize; ++i) {
        biasGradients[i] = incomingGradient[i];
    }

    // dL/dx = W^T * dL/dz
    std::vector<double> inputGradientData(inputSize, 0.0);

    for (size_t j = 0; j < inputSize; ++j) {
        double sum = 0.0;

        for (size_t i = 0; i < outputSize; ++i) {
            sum += weights(i, j) * incomingGradient[i];
        }

        inputGradientData[j] = sum;
    }

    return Vector(std::move(inputGradientData));
}

void DenseLayer::applyGradients(double learningRate) {
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }

    for (size_t i = 0; i < weights.rowCount(); ++i) {
        for (size_t j = 0; j < weights.colCount(); ++j) {
            weights(i, j) -= learningRate * weightGradients(i, j);
        }
    }

    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] -= learningRate * biasGradients[i];
    }
}

}
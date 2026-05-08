#pragma once

#include "ml/core/Matrix.h"
#include "ml/core/Vector.h"

namespace machineLearning {

class DenseLayer {
private: 
    Matrix weights;
    Vector bias;

    Matrix weightGradients;
    Vector biasGradients;

    Vector inputCache;

public:
    DenseLayer(size_t inputSize, size_t outputSize);

    const Matrix& getWeights() const;
    const Vector& getBias() const;

    Vector forward(const Vector& input);
    Vector backward(const Vector& incomingGradient);

    void applyGradients(double learningRate);
};
}
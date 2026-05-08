#pragma once

#include "ml/core/Vector.h"

namespace machineLearning {

class SigmoidActivation {
private:
    Vector inputCache;

public:
    explicit SigmoidActivation(size_t inputSize);

    Vector forward(const Vector& input);
    Vector backward(const Vector& incomingGradient);
};
}
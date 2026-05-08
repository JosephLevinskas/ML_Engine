#pragma once

#include "ml/core/Vector.h"

namespace machineLearning {

class ReLUActivation {
private:
    Vector inputCache;

public:
    explicit ReLUActivation(size_t inputSize);

    Vector forward(const Vector& input);
};
}
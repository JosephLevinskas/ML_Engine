#pragma once

#include "ml/core/Vector.h"

namespace machineLearning {

double meanSquaredError(const Vector& predictions, const Vector& targets);
double binaryCrossEntropy(const Vector& probabilities, const Vector& targets);
}

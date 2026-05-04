#pragma once

#include "ml/Vector.h"

namespace machineLearning {

double meanSquaredError(const Vector& predictions, const Vector& targets);
}
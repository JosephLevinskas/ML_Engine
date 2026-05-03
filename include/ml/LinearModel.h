#pragma once

#include "ml/Vector.h"
#include "ml/Matrix.h"

namespace machineLearning {

class LinearModel {
private:
    Vector weights;
    double bias;

public:
    LinearModel(Vector weights_, double bias);

    double predict(const Vector& x) const;
    Vector predict(const Matrix& x) const;
};
}
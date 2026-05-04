#pragma once

#include "ml/Vector.h"
#include "ml/Matrix.h"

#include <optional>

namespace machineLearning {

class StandardScaler {
private:
    Vector means; // Per feature
    Vector stdDevs; // Per feature
    bool fitted = false;

public:
    StandardScaler(Vector means_, Vector stdDevs_, bool fitted_);

    void fit(const Matrix& X);
    Matrix transform(const Matrix& X) const;
    Matrix fitTransform(const Matrix& X);
};
}
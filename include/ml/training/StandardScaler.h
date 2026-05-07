#pragma once

#include "ml/core/Vector.h"
#include "ml/core/Matrix.h"

#include <optional>

namespace machineLearning {

class StandardScaler {
private:
    std::optional<Vector> means; // Per feature
    std::optional<Vector> stdDevs; // Per feature
    bool fitted = false;

public:
    StandardScaler(Vector means_, Vector stdDevs_, bool fitted_);
    StandardScaler() = default;

    void fit(const Matrix& X);
    Matrix transform(const Matrix& X) const;
    Matrix fitTransform(const Matrix& X);
};
}

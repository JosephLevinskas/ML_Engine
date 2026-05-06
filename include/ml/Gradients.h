#pragma once

#include "ml/Vector.h"
#include "ml/Matrix.h"


#include <vector>

namespace machineLearning {

struct LinearModelGradients {
    Vector weightGradients;
    double biasGradient;
};

struct LogisticModelGradients {
    Vector weightGradients;
    double biasGradient;
};

LinearModelGradients computeLinearModelMSEGradients(const Matrix& X, 
                        const Vector& predictions, const Vector& targets);

LogisticModelGradients computeLogisticModelBCEGradients(
    const Matrix& X,
    const Vector& probabilities,
    const Vector& targets
);

}
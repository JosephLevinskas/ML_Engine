#include "ml/LinearModel.h"
#include "ml/Matrix.h"
#include "ml/Vector.h"
#include "ml/Operations.h"

#include <utility>
#include <stdexcept>

namespace machineLearning {

LinearModel::LinearModel(Vector weights_, double bias_)
    : weights(std::move(weights_)), bias(bias_)
{
    if (weights.size() == 0) {
        throw std::invalid_argument("Linear model can't have an empty Vector");
    }
}

const Vector& LinearModel::getWeights() const {
    return weights;
}

const double LinearModel::getBias() const {
    return bias;
}

double LinearModel::predict(const Vector& x) const{
    if (x.size() != weights.size()) {
        throw std::invalid_argument("Input vector size must match weight size");
    }
    
    return dot(x, weights) + bias;
}

Vector LinearModel::predict(const Matrix& x) const {
    if (x.colCount() != weights.size()) {
        throw std::invalid_argument("Matrix column count must match weight size");
    }

    return (x * weights) + bias;
}
}
#include "ml/LossFunctions.h"
#include "ml/Vector.h"
#include "ml/Operations.h"

#include <utility>
#include <stdexcept>

namespace machineLearning {

double meanSquaredError(const Vector& predictions, const Vector& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must be same size for error calculatrion");
    }
    
    Vector resultV = predictions - targets;
    double resultD = dot(resultV, resultV) / targets.size();
    return resultD;
    
}
}
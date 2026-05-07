#include "ml/training/LossFunctions.h"
#include "ml/core/Vector.h"
#include "ml/core/Operations.h"

#include <utility>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace machineLearning {

double meanSquaredError(const Vector& predictions, const Vector& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must be same size for error calculatrion");
    }
    
    Vector resultV = predictions - targets;
    double resultD = dot(resultV, resultV) / targets.size();
    return resultD;
    
}

double binaryCrossEntropy(const Vector& probabilities, const Vector& targets) {
    if (probabilities.size() != targets.size()) {
        throw std::invalid_argument("Probabilities and targets must be same size for binary cross entropy");
    }

    constexpr double epsilon = 1e-15;
    double loss = 0.0;

    for (size_t i = 0; i < probabilities.size(); ++i) {
        double p = probabilities[i];
        double y = targets[i];

        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Probability must be between 0 and 1");
        }

        if (y != 0.0 && y != 1.0) {
            throw std::invalid_argument("Target must be 0 or 1 for binary cross entropy");
        }

        p = std::clamp(p, epsilon, 1.0 - epsilon);

        loss += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    }

    return loss / probabilities.size();
}


double l2Penalty(const Vector& weights, double lambda) {
    double sum = 0.0;

    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * weights[i];
    }

    return 0.5 * lambda * sum;
}
}

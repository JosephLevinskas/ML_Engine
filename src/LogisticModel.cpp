#include "ml/LogisticModel.h"
#include "ml/Operations.h"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace machineLearning {

LogisticModel::LogisticModel(Vector weights_, double bias_) 
    : weights(std::move(weights_)), bias(bias_)
{}

const Vector& LogisticModel::getWeights() const {
    return weights;
}

double LogisticModel::getBias() const {
    return bias;
}

double LogisticModel::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double LogisticModel::predictProbability(const Vector& X) const {
    if (X.size() != weights.size()) {
        throw std::invalid_argument("Size of vector and weights must be the same to take dot product");
    }
    double z = dot(X, weights) + bias;
    return sigmoid(z);
}

Vector LogisticModel::predictProbability(const Matrix& X) const {
    std::vector<double> probabilities;
    probabilities.reserve(X.rowCount());
    
    size_t rowC = X.rowCount();
    size_t colC = X.colCount();

    for (size_t i = 0; i < rowC; ++i) {
        std::vector<double> rowData;
        rowData.reserve(colC);

        for (size_t j = 0; j < colC; ++j) {
            rowData.push_back(X(i, j));
        }
        Vector row(std::move(rowData));
        probabilities.push_back(predictProbability(row));
    }
    return Vector(std::move(probabilities));
}

double LogisticModel::predictClass(const Vector& X) const {
    const double result = predictProbability(X);
    if (result >= 0.5) {
        return 1.0;
    }
    else {
        return 0.0;
    }
}

Vector LogisticModel::predictClass(const Matrix& X) const {
    std::vector<double> results;
    results.reserve(X.rowCount());

    Vector probabilities = predictProbability(X);

    size_t rowC = probabilities.size();
    for (size_t i = 0; i < rowC; ++i) {
        if (probabilities[i] >= 0.5) {
            results.push_back(1.0);
        }
        else {
            results.push_back(0.0);
        }
    }
    return Vector(results);
}

}
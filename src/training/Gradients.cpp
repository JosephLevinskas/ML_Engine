#include "ml/training/Gradients.h"

#include <stdexcept>
#include <utility>


namespace machineLearning {

LinearModelGradients computeLinearModelMSEGradients(const Matrix& X, 
                        const Vector& predictions, const Vector& targets) {
        const size_t rowM = X.rowCount();
        const size_t colM = X.colCount();
        
        if (rowM != predictions.size() || rowM != targets.size()) {
            throw std::invalid_argument("Invalid size for linear model gradients input");
        }

        std::vector<double> weightGradientResults(colM, 0.0);
        Vector error = predictions - targets;

        for (size_t j = 0; j < colM; ++j) {
            double errorSum = 0.0;
            for (size_t i = 0; i < rowM; ++i) {
                errorSum +=  error[i] * X(i, j);
            }
            weightGradientResults[j] = (2.0 / rowM) * errorSum;
        }


        double biasErrorSum = 0.0;
        for (size_t i = 0; i < rowM; ++i) {
            biasErrorSum += error[i];
        }
        
        const double biasGradientResult = (2.0 / rowM) * biasErrorSum;
        
        
        return LinearModelGradients{Vector(std::move(weightGradientResults)), biasGradientResult};
    }

LogisticModelGradients computeLogisticModelBCEGradients(
    const Matrix& X,
    const Vector& probabilities,
    const Vector& targets
) {
    const size_t rowC = X.rowCount();
    const size_t colC = X.colCount();

    if (rowC != probabilities.size() || rowC != targets.size()) {
        throw std::invalid_argument("Invalid size for logistic model gradients input");
    }

    std::vector<double> weightResults(colC, 0.0);
    double biasGrad = 0.0;

    for (size_t i = 0; i < rowC; ++i) {
        const double error = probabilities[i] - targets[i];

        for (size_t j = 0; j < colC; ++j) {
            weightResults[j] += error * X(i, j);
        }

        biasGrad += error;
    }

    for (double& grad : weightResults) {
        grad /= rowC;
    }

    biasGrad /= rowC;

    return LogisticModelGradients{Vector(std::move(weightResults)), biasGrad};
}  
}

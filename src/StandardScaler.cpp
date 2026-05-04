#include "ml/StandardScaler.h"

#include <stdexcept>
#include <vector>
#include <cmath>

namespace machineLearning {

StandardScaler::StandardScaler(Vector means_, Vector stdDevs_, bool fitted_)
    : means(std::move(means_)),
      stdDevs(std::move(stdDevs_)),
      fitted(fitted_)
{}

void StandardScaler::fit(const Matrix& X) {
    const size_t colC = X.colCount();
    const size_t rowC = X.rowCount();

    std::vector<double> meanData(X.colCount(), 0.0);
    std::vector<double> stdData(X.colCount(), 0.0);

    for (size_t j = 0; j < colC; ++j) {
        double sum = 0.0;

        for (size_t i = 0; i < rowC; ++i) {
            sum += X(i, j);
        }

        meanData[j] = sum / static_cast<double>(rowC);
    }

    for (size_t j = 0; j < colC; ++j) {
        double squaredDiffSum = 0.0;
        const double colMean = meanData[j];

        for (size_t i = 0; i < rowC; ++i) {
            double temp = X(i, j) - colMean;
            temp = temp * temp;
            squaredDiffSum += temp;
        }

        const double variance = squaredDiffSum / static_cast<double>(rowC);
        double std = std::sqrt(variance);
        
        if (std == 0.0) {
            std = 1.0;
        }

        stdData[j] = std;
    }

    means = Vector(std::move(meanData));
    stdDevs = Vector(std::move(stdData));
    fitted = true;
}
Matrix StandardScaler::transform(const Matrix& X) const {
    if (!fitted) {
        throw std::invalid_argument("Data not fitted yet, cannot scale");
    }

    if (X.colCount() != means.size() || X.colCount() != stdDevs.size()) {
        throw std::invalid_argument("Mean and stdDevs needs to be the same size as column count");
    }

    const size_t rowC = X.rowCount();
    const size_t colC = X.colCount();


    Matrix newMatrix = X;
    for (size_t i = 0; i < rowC; ++i) {
        for (size_t j = 0; j < colC; ++j) {
            newMatrix(i, j) = (X(i, j) - means[j]) / stdDevs[j];
        }
    }
    return newMatrix;
}

Matrix StandardScaler::fitTransform(const Matrix& X) {
    fit(X);
    return transform(X);
}

}


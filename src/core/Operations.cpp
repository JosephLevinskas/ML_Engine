#include "ml/core/Vector.h"
#include "ml/core/Matrix.h"

#include <stdexcept>
#include <utility>

namespace machineLearning {

double dot(const Vector& v1, const Vector& v2) {
    const size_t size = v1.size();

    if (size != v2.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }

    double sum = 0.0;

    for (size_t i = 0; i < size; ++i) {
        sum += v1[i] * v2[i];
    }

    return sum;

}

Vector operator*(const Matrix& m, const Vector& v) {
    const size_t mRowCount = m.rowCount();
    const size_t mColCount = m.colCount();

    if (mColCount != v.size()) {
        throw std::invalid_argument("Size mismatch can't do matrix-vector multiplication");
    }

    std::vector<double> resultV(mRowCount);
    
    for (size_t i = 0; i < mRowCount; ++i) {
        double sum = 0.0;
        
        for (size_t j = 0; j < mColCount; ++j) {
            sum += m(i, j) * v[j];
        }

        resultV[i] = sum;

    }
    
    return Vector(std::move(resultV));
}
}

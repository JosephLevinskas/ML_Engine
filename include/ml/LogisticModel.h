#pragma once

#include "ml/Vector.h"
#include "ml/Matrix.h"

namespace machineLearning {

class LogisticModel {
private:
    Vector weights;
    double bias;

    static double sigmoid(double z);

public:
    LogisticModel(Vector weights_, double bias_);

    double predictProbability(const Vector& X) const;
    Vector predictProbability(const Matrix& X) const;

    double predictClass(const Vector& X) const;
    Vector predictClass(const Matrix& X) const;

    const Vector& getWeights() const;
    double getBias() const;

}; 
}
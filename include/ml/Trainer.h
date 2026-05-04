#pragma once

#include "ml/LinearModel.h"
#include "ml/Matrix.h"
#include "ml/Vector.h"

namespace machineLearning {

struct TrainingResults {
    LinearModel model;
    std::vector<double> losses;
};

class Trainer {
private:
    double learningRate;
    size_t epochs;

public:
    Trainer(double learningRate_, size_t epochs_);

    TrainingResults train(const LinearModel& model,
                        const Matrix& X,
                        const Vector& Targets) const;
};
}
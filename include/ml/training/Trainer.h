#pragma once

#include "ml/models/LinearModel.h"
#include "ml/core/Matrix.h"
#include "ml/core/Vector.h"
#include "ml/models/LogisticModel.h"
namespace machineLearning {

struct TrainingResults {
    LinearModel model;
    std::vector<double> losses;
};

struct LogisticTrainingResults {
    LogisticModel model;
    std::vector<double> losses;
};

class Trainer {
private:
    double learningRate;
    size_t epochs;

public:
    Trainer(double learningRate_, size_t epochs_);

    TrainingResults train(
        const LinearModel& model,
        const Matrix& X,
        const Vector& targets
    ) const;

    LogisticTrainingResults train(
        const LogisticModel& model,
        const Matrix& X,
        const Vector& targets
    ) const;
};
}

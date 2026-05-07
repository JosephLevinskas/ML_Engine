#pragma once

#include "ml/models/LinearModel.h"
#include "ml/core/Matrix.h"
#include "ml/core/Vector.h"
#include "ml/training/StandardScaler.h"
#include "ml/training/Trainer.h"

namespace machineLearning {

class LinearRegressionPipeline {
private:
    StandardScaler scaler;
    LinearModel model;
    bool trained = false;

public:
    LinearRegressionPipeline(const LinearModel& initialModel);
    TrainingResults train(const Matrix& X, const Vector& y, Trainer& trainer);
    Vector predict(const Matrix& X) const;
};
}

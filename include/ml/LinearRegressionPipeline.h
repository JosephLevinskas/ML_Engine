#pragma once

#include "ml/LinearModel.h"
#include "ml/Matrix.h"
#include "ml/Vector.h"
#include "ml/StandardScaler.h"
#include "ml/Trainer.h"

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
#pragma once

#include "ml/data/DataSetLoader.h"
#include "ml/models/LogisticModel.h"
#include "ml/training/StandardScaler.h"
#include "ml/training/Trainer.h"
#include "ml/core/Vector.h"
#include "ml/core/Matrix.h"

namespace machineLearning {

class LogisticRegressionPipeline {
private:
    StandardScaler scaler;
    LogisticModel model;
    bool trained;

public:
    LogisticRegressionPipeline(const LogisticModel& initialModel);

    LogisticTrainingResults train(
        const Dataset& dataset,
        Trainer& trainer
    );

    Vector predictProbability(const Matrix& X) const;

    Vector predictClass(const Matrix& X) const;

    const LogisticModel& getModel() const;
};

}

#pragma once

#include "ml/DataSetLoader.h"
#include "ml/LogisticModel.h"
#include "ml/StandardScaler.h"
#include "ml/Trainer.h"
#include "ml/Vector.h"
#include "ml/Matrix.h"

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
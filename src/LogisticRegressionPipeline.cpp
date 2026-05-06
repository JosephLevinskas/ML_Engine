#include "ml/LogisticRegressionPipeline.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace machineLearning {

LogisticRegressionPipeline::LogisticRegressionPipeline(
    const LogisticModel& initialModel
)
    : scaler(),
      model(initialModel),
      trained(false)
{}

LogisticTrainingResults LogisticRegressionPipeline::train(
    const Dataset& dataset,
    Trainer& trainer
) {
    Matrix scaledX = scaler.fitTransform(dataset.features);

    LogisticTrainingResults results =
        trainer.train(model, scaledX, dataset.targets);

    model = std::move(results.model);
    trained = true;

    return LogisticTrainingResults{model, std::move(results.losses)};
}

Vector LogisticRegressionPipeline::predictProbability(const Matrix& X) const {
    if (!trained) {
        throw std::logic_error("Pipeline must be trained before prediction");
    }

    Matrix scaledX = scaler.transform(X);
    return model.predictProbability(scaledX);
}

Vector LogisticRegressionPipeline::predictClass(const Matrix& X) const {
    if (!trained) {
        throw std::logic_error("Pipeline must be trained before prediction");
    }

    Matrix scaledX = scaler.transform(X);
    return model.predictClass(scaledX);
}

const LogisticModel& LogisticRegressionPipeline::getModel() const {
    return model;
}

}
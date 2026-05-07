#include "ml/training/Trainer.h"
#include "ml/training/Gradients.h"
#include "ml/training/LossFunctions.h"

#include <utility>
#include <stdexcept>

namespace machineLearning {

Trainer::Trainer(double learningRate_, size_t epochs_)
    : learningRate(learningRate_), epochs(epochs_), lambda(0.0) 
{
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (epochs == 0) {
        throw std::invalid_argument("Epoch count must be greater than zero");
    }
}

Trainer::Trainer(double learningRate_, size_t epochs_, double lambda_)
    : learningRate(learningRate_), epochs(epochs_), lambda(lambda_) 
{
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (epochs == 0) {
        throw std::invalid_argument("Epoch count must be greater than zero");
    }
    if (lambda < 0.0) {
        throw std::invalid_argument("Lambda must be non-negative");
    }
}

TrainingResults Trainer::train(const LinearModel& model, const Matrix& X, const Vector& targets) const {
    LinearModel current = model;
    std::vector<double> dataLosses;
    dataLosses.reserve(epochs);
    std::vector<double> objectiveLosses;
    objectiveLosses.reserve(epochs);
    
    for (size_t i = 0; i < epochs; ++i) {
        Vector predictions = current.predict(X);
        double dataLoss = meanSquaredError(predictions, targets);
        double objectiveLoss = dataLoss + l2Penalty(current.getWeights(), lambda);

        dataLosses.push_back(dataLoss);
        objectiveLosses.push_back(objectiveLoss);
        LinearModelGradients gradients = computeLinearModelMSEGradients(X, predictions, targets);
        
        for (size_t j = 0; j < gradients.weightGradients.size(); ++j) {
            gradients.weightGradients[j] += lambda * current.getWeights()[j];
        }
        
        Vector updatedWeights = (current.getWeights() - (learningRate * gradients.weightGradients));
        double updatedBias = (current.getBias() - (learningRate * gradients.biasGradient));
        current = LinearModel(std::move(updatedWeights), updatedBias);
    }
    return {std::move(current), std::move(dataLosses), std::move(objectiveLosses)};
}

LogisticTrainingResults Trainer::train(const LogisticModel& model, const Matrix& X, const Vector& targets) const {
    LogisticModel current = model;
    std::vector<double> dataLosses;
    dataLosses.reserve(epochs);
    std::vector<double> objectiveLosses;
    objectiveLosses.reserve(epochs);

    for (size_t i = 0; i < epochs; ++i) {
        Vector probabilities = current.predictProbability(X);
        double dataLoss = binaryCrossEntropy(probabilities, targets);
        double objectiveLoss = dataLoss + l2Penalty(current.getWeights(), lambda);

        dataLosses.push_back(dataLoss);
        objectiveLosses.push_back(objectiveLoss);
        LogisticModelGradients gradients = computeLogisticModelBCEGradients(X, probabilities, targets);
        
        for (size_t j = 0; j < gradients.weightGradients.size(); ++j) {
            gradients.weightGradients[j] += lambda * current.getWeights()[j];
        }
        
        Vector updatedWeights = (current.getWeights() - (learningRate * gradients.weightGradients));
        double updatedBias = (current.getBias() - (learningRate * gradients.biasGradient));
        current = LogisticModel(std::move(updatedWeights), updatedBias);

    }
    return {std::move(current), std::move(dataLosses), std::move(objectiveLosses)};

}

}

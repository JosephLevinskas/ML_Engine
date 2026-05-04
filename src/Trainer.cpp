#include "ml/Trainer.h"
#include "ml/Gradients.h"

#include <utility>
#include <stdexcept>

namespace machineLearning {

Trainer::Trainer(double learningRate_, size_t epochs_)
    : learningRate(learningRate_), epochs(epochs_)
{
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (epochs == 0) {
        throw std::invalid_argument("Epoch count must be greater than zero");
    }
}

LinearModel Trainer::train(const LinearModel& model, const Matrix& X, const Vector& targets) const {
    LinearModel current = model;
    
    for (size_t i = 0; i < epochs; ++i) {
        Vector predictions = current.predict(X);
        LinearModelGradients gradients = computeLinearModelMSEGradients(X, predictions, targets);
        Vector updatedWeights = (current.getWeights() - (learningRate * gradients.weightGradients));
        double updatedBias = (current.getBias() - (learningRate * gradients.biasGradient));
        current = LinearModel(std::move(updatedWeights), updatedBias);
    }
    return current;
}

}
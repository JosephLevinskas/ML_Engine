#include "ml/Matrix.h"
#include "ml/Vector.h"
#include "ml/LinearModel.h"
#include "ml/LossFunctions.h"
#include "ml/Gradients.h"
#include "ml/Operations.h"

#include <iostream>

using namespace machineLearning;

int main() {
    // 1. Create dataset (2 samples, 2 features)
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Vector targets({8.0, 18.0});

    // 2. Initialize model
    // Let's say: prediction = 1*x1 + 2*x2 + 0
    LinearModel model(Vector({1.0, 2.0}), 0.0);

    // 3. Forward pass
    Vector predictions = model.predict(X);

    std::cout << "Predictions:\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << predictions[i] << "\n";
    }

    // 4. Compute loss
    double loss = meanSquaredError(predictions, targets);
    std::cout << "\nMSE Loss: " << loss << "\n";

    // 5. Compute gradients
    LinearModelGradients grads =
        computeLinearModelMSEGradients(X, predictions, targets);

    std::cout << "\nWeight Gradients:\n";
    for (size_t i = 0; i < grads.weightGradients.size(); ++i) {
        std::cout << grads.weightGradients[i] << "\n";
    }

    std::cout << "\nBias Gradient:\n";
    std::cout << grads.biasGradient << "\n";

    const double learningRate = 0.01;

    Vector updatedWeights =
        Vector({1.0, 2.0}) - (grads.weightGradients * learningRate);

    double updatedBias =
        0.0 - learningRate * grads.biasGradient;

    LinearModel updatedModel(updatedWeights, updatedBias);

    Vector updatedPredictions = updatedModel.predict(X);

    double updatedLoss = meanSquaredError(updatedPredictions, targets);

    std::cout << "\nUpdated Predictions:\n";
    for (size_t i = 0; i < updatedPredictions.size(); ++i) {
        std::cout << updatedPredictions[i] << "\n";
    }

    std::cout << "\nUpdated MSE Loss:\n";
    std::cout << updatedLoss << "\n";

    return 0;
}
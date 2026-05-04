#include "ml/Matrix.h"
#include "ml/Vector.h"
#include "ml/LinearModel.h"
#include "ml/LossFunctions.h"
#include "ml/Trainer.h"

#include <iostream>
#include <vector>

using namespace machineLearning;

int main() {
    // --- TRUE MODEL ---
    Vector trueWeights({3.0, 5.0});
    double trueBias = 2.0;

    // --- DATASET ---
    const size_t rows = 1000;
    const size_t cols = 2;

    std::vector<double> data;
    data.reserve(rows * cols);

    std::vector<double> targetData;
    targetData.reserve(rows);

    for (size_t i = 0; i < rows; ++i) {
        double x1 = static_cast<double>(i) / 1000.0;
        double x2 = static_cast<double>(i % 10) / 10.0;

        data.push_back(x1);
        data.push_back(x2);

        double y = x1 * trueWeights[0] + x2 * trueWeights[1] + trueBias;
        targetData.push_back(y);
    }

    Matrix X(rows, cols, std::move(data));
    Vector targets(std::move(targetData));

    // --- INITIAL MODEL ---
    LinearModel model(Vector({0.1, 0.1}), 0.0);

    std::cout << "Initial Weights: "
              << model.getWeights()[0] << ", "
              << model.getWeights()[1] << "\n";

    std::cout << "Initial Bias: " << model.getBias() << "\n";

    double initialLoss = meanSquaredError(model.predict(X), targets);
    std::cout << "Initial Loss: " << initialLoss << "\n";

    // --- TRAIN ---
    Trainer trainer(0.1, 1000);

    TrainingResults result = trainer.train(model, X, targets);
    const LinearModel& trained = result.model;

    // --- RESULTS ---
    std::cout << "\nTrained Weights: "
              << trained.getWeights()[0] << ", "
              << trained.getWeights()[1] << "\n";

    std::cout << "Trained Bias: " << trained.getBias() << "\n";

    double finalLoss = meanSquaredError(trained.predict(X), targets);
    std::cout << "Final Loss: " << finalLoss << "\n";

    // --- LOSS PROGRESSION ---
    std::cout << "\nLoss progression:\n";
    std::cout << "Start: " << result.losses.front() << "\n";
    std::cout << "End:   " << result.losses.back() << "\n";

    // Optional: print every 100 epochs
    std::cout << "\nSampled Losses:\n";
    for (size_t i = 0; i < result.losses.size(); i += 100) {
        std::cout << "Epoch " << i << ": " << result.losses[i] << "\n";
    }

    // --- TRUE VALUES ---
    std::cout << "\nTrue Weights: "
              << trueWeights[0] << ", "
              << trueWeights[1] << "\n";

    std::cout << "True Bias: " << trueBias << "\n";

    return 0;
}
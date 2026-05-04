#include "ml/Matrix.h"
#include "ml/Vector.h"
#include "ml/LinearModel.h"
#include "ml/LossFunctions.h"
#include "ml/Trainer.h"
#include "ml/StandardScaler.h"

#include <iostream>
#include <vector>

using namespace machineLearning;

int main() {
    std::cout << "=== ML ENGINE DEMO ===\n\n";

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

    // --- SCALER (WORKAROUND CONSTRUCTOR) ---
    StandardScaler scaler(
        Vector({0.0, 0.0}),   // dummy means
        Vector({1.0, 1.0}),   // dummy stds
        false                 // not fitted yet
    );

    Matrix scaledX = scaler.fitTransform(X);

    std::cout << "Scaling complete.\n";
    std::cout << "Sample scaled row: "
              << scaledX(0, 0) << ", "
              << scaledX(0, 1) << "\n\n";

    // --- INITIAL MODEL ---
    LinearModel model(Vector({0.1, 0.1}), 0.0);

    std::cout << "Initial Weights: "
              << model.getWeights()[0] << ", "
              << model.getWeights()[1] << "\n";

    std::cout << "Initial Bias: " << model.getBias() << "\n";

    double initialLoss = meanSquaredError(model.predict(scaledX), targets);
    std::cout << "Initial Loss: " << initialLoss << "\n\n";

    // --- TRAIN ---
    Trainer trainer(0.1, 1000);
    TrainingResults result = trainer.train(model, scaledX, targets);
    const LinearModel& trained = result.model;

    // --- RESULTS ---
    std::cout << "=== TRAINING COMPLETE ===\n\n";

    std::cout << "Trained Weights: "
              << trained.getWeights()[0] << ", "
              << trained.getWeights()[1] << "\n";

    std::cout << "Trained Bias: " << trained.getBias() << "\n";

    double finalLoss = meanSquaredError(trained.predict(scaledX), targets);
    std::cout << "Final Loss: " << finalLoss << "\n\n";

    std::cout << "Loss progression:\n";
    std::cout << "Start: " << result.losses.front() << "\n";
    std::cout << "End:   " << result.losses.back() << "\n\n";

    std::cout << "True Weights: "
              << trueWeights[0] << ", "
              << trueWeights[1] << "\n";

    std::cout << "True Bias: " << trueBias << "\n";

    std::cout << "\nVerifying predictions against true values:\n";

    for (int i = 0; i < 5; ++i) {
        Vector row({X(i, 0), X(i, 1)});
        Vector scaledRow = scaler.transform(Matrix(1, 2, {X(i,0), X(i,1)})).getRow(0);

        double pred = trained.predict(scaledRow);
        double trueVal = X(i,0) * 3 + X(i,1) * 5 + 2;

        std::cout << "Pred: " << pred << " | True: " << trueVal << "\n";
    }

    return 0;
}
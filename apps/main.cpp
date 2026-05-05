#include "ml/Matrix.h"
#include "ml/Vector.h"
#include "ml/LinearModel.h"
#include "ml/Trainer.h"
#include "ml/LossFunctions.h"
#include "ml/LinearRegressionPipeline.h"

#include <iostream>
#include <vector>

using namespace machineLearning;

int main() {
    std::cout << "=== LINEAR REGRESSION PIPELINE DEMO ===\n\n";

    // --- TRUE MODEL ---
    Vector trueWeights({3.0, 5.0});
    double trueBias = 2.0;

    // --- DATASET ---
    const size_t rows = 1000;
    const size_t cols = 2;

    std::vector<double> data;
    std::vector<double> targetData;

    data.reserve(rows * cols);
    targetData.reserve(rows);

    for (size_t i = 0; i < rows; ++i) {
        double x1 = static_cast<double>(i) / 1000.0;
        double x2 = static_cast<double>(i % 10) / 10.0;

        data.push_back(x1);
        data.push_back(x2);

        double y = 3.0 * x1 + 5.0 * x2 + 2.0;
        targetData.push_back(y);
    }

    Matrix X(rows, cols, std::move(data));
    Vector y(std::move(targetData));

    std::cout << "Dataset created.\n";
    std::cout << "Rows: " << X.rowCount() << ", Cols: " << X.colCount() << "\n\n";

    // --- INITIAL MODEL ---
    LinearModel initialModel(Vector({0.1, 0.1}), 0.0);

    std::cout << "Initial Model:\n";
    std::cout << "Weights: "
              << initialModel.getWeights()[0] << ", "
              << initialModel.getWeights()[1] << "\n";
    std::cout << "Bias: " << initialModel.getBias() << "\n\n";

    // --- PIPELINE ---
    LinearRegressionPipeline pipeline(initialModel);

    Trainer trainer(0.1, 1000);

    std::cout << "Training pipeline...\n\n";

    TrainingResults result = pipeline.train(X, y, trainer);

    // --- TRAINING RESULTS ---
    std::cout << "=== TRAINING RESULTS ===\n";
    std::cout << "Initial Loss: " << result.losses.front() << "\n";
    std::cout << "Final Loss:   " << result.losses.back() << "\n\n";

    std::cout << "Sample Loss Progression:\n";
    for (size_t i = 0; i < result.losses.size(); i += 200) {
        std::cout << "Epoch " << i << ": " << result.losses[i] << "\n";
    }
    std::cout << "\n";

    // --- PREDICTIONS ON TRAINING DATA ---
    Vector predictions = pipeline.predict(X);
    double trainLoss = meanSquaredError(predictions, y);

    std::cout << "Prediction Loss on Training Data: " << trainLoss << "\n\n";

    // --- SAMPLE PREDICTIONS ---
    std::cout << "Sample Predictions vs True Values:\n";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "Pred: " << predictions[i]
                  << " | True: " << y[i] << "\n";
    }
    std::cout << "\n";

    // --- TEST ON NEW DATA ---
    Matrix newX(3, 2, {
        0.25, 0.5,
        0.75, 1.2,
        1.5,  0.3
    });

    Vector newPreds = pipeline.predict(newX);

    std::cout << "Predictions on New Data:\n";
    for (size_t i = 0; i < newPreds.size(); ++i) {
        double trueVal =
            3.0 * newX(i, 0) +
            5.0 * newX(i, 1) +
            2.0;

        std::cout << "Input: (" << newX(i, 0) << ", " << newX(i, 1) << ") "
                  << "| Pred: " << newPreds[i]
                  << " | True: " << trueVal << "\n";
    }

    std::cout << "\n=== TRUE MODEL ===\n";
    std::cout << "Weights: 3, 5\n";
    std::cout << "Bias: 2\n";

    return 0;
}
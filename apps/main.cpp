#include "ml/DataSetLoader.h"
#include "ml/LogisticModel.h"
#include "ml/Trainer.h"
#include "ml/StandardScaler.h"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace machineLearning;

Vector rowToVector(const Matrix& matrix, size_t row) {
    std::vector<double> data;
    data.reserve(matrix.colCount());

    for (size_t j = 0; j < matrix.colCount(); ++j) {
        data.push_back(matrix(row, j));
    }

    return Vector(std::move(data));
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    try {
        const std::string path = "assets/User_Data.csv";
        const size_t targetColumn = 2;

        std::cout << "Loading dataset: " << path << "\n";

        Dataset dataset = DataSetLoader::loadCSV(path, targetColumn);

        std::cout << "Dataset loaded.\n";
        std::cout << "Samples:  " << dataset.features.rowCount() << "\n";
        std::cout << "Features: " << dataset.features.colCount() << "\n\n";

        StandardScaler scaler;
        Matrix scaledX = scaler.fitTransform(dataset.features);

        Vector initialWeights(std::vector<double>(scaledX.colCount(), 0.0));
        LogisticModel initialModel(initialWeights, 0.0);

        Trainer trainer(0.1, 5000);

        std::cout << "Training logistic regression with scaled features...\n";

        LogisticTrainingResults results =
            trainer.train(initialModel, scaledX, dataset.targets);

        std::cout << "Training complete.\n\n";

        std::cout << "Initial loss: " << results.losses.front() << "\n";
        std::cout << "Final loss:   " << results.losses.back() << "\n\n";

        size_t correct = 0;

        std::cout << "First 20 predictions:\n";

        const size_t rowsToPrint =
            std::min<size_t>(20, dataset.features.rowCount());

        for (size_t i = 0; i < dataset.features.rowCount(); ++i) {
            Vector x = rowToVector(scaledX, i);

            double probability = results.model.predictProbability(x);
            int prediction = results.model.predictClass(x);
            int target = static_cast<int>(dataset.targets[i]);

            if (prediction == target) {
                ++correct;
            }

            if (i < rowsToPrint) {
                std::cout << "Sample " << i
                          << " | age = " << dataset.features(i, 0)
                          << " | salary = " << dataset.features(i, 1)
                          << " | prob = " << probability
                          << " | pred = " << prediction
                          << " | target = " << target
                          << "\n";
            }
        }

        double accuracy =
            static_cast<double>(correct) / dataset.features.rowCount();

        std::cout << "\nAccuracy: " << accuracy * 100.0 << "%\n\n";

        std::cout << "Manual predictions:\n";

        Matrix manualRaw(
            4,
            2,
            {
                25.0, 30000.0,
                30.0, 140000.0,
                50.0, 60000.0,
                55.0, 120000.0
            }
        );

        Matrix manualScaled = scaler.transform(manualRaw);

        for (size_t i = 0; i < manualScaled.rowCount(); ++i) {
            Vector x = rowToVector(manualScaled, i);

            std::cout << "Age " << manualRaw(i, 0)
                      << ", Salary " << manualRaw(i, 1)
                      << " -> prob = "
                      << results.model.predictProbability(x)
                      << " | class = "
                      << results.model.predictClass(x)
                      << "\n";
        }

        std::cout << "\nDone.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
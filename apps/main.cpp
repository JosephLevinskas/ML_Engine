#include "ml/DataSetLoader.h"
#include "ml/DataSplitter.h"
#include "ml/LogisticModel.h"
#include "ml/Trainer.h"
#include "ml/StandardScaler.h"

#include <iostream>
#include <vector>

using namespace machineLearning;

Vector rowToVector(const Matrix& m, size_t row) {
    std::vector<double> data;
    data.reserve(m.colCount());

    for (size_t j = 0; j < m.colCount(); ++j) {
        data.push_back(m(row, j));
    }

    return Vector(std::move(data));
}

int main() {
    try {
        // ---- Load dataset ----
        Dataset dataset =
            DataSetLoader::loadCSV("assets/User_Data.csv", 2);

        // ---- Split dataset ----
        DatasetSplit split =
            DataSplitter::split(dataset, 0.8);

        // ---- Scale (IMPORTANT: fit only on train) ----
        StandardScaler scaler;

        Matrix trainX = scaler.fitTransform(split.train.features);
        Matrix testX  = scaler.transform(split.test.features);

        // ---- Model ----
        Vector initialWeights(
            std::vector<double>(trainX.colCount(), 0.0)
        );

        LogisticModel model(initialWeights, 0.0);

        // ---- Train ----
        Trainer trainer(0.1, 5000);

        LogisticTrainingResults results =
            trainer.train(model, trainX, split.train.targets);

        std::cout << "Train loss: "
                  << results.losses.back() << "\n";

        // ---- Evaluate on TEST set ----
        size_t correct = 0;

        for (size_t i = 0; i < testX.rowCount(); ++i) {
            Vector x = rowToVector(testX, i);

            int pred = results.model.predictClass(x);
            int target = static_cast<int>(split.test.targets[i]);

            if (pred == target) {
                ++correct;
            }
        }

        double accuracy =
            static_cast<double>(correct) / testX.rowCount();

        std::cout << "Test Accuracy: "
                  << accuracy * 100.0 << "%\n";
    }
    catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
    }

    return 0;
}
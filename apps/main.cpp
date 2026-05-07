#include "ml/data/DataSetLoader.h"
#include "ml/data/DataSplitter.h"
#include "ml/models/LogisticModel.h"
#include "ml/pipelines/LogisticRegressionPipeline.h"
#include "ml/training/Trainer.h"

#include <exception>
#include <iostream>
#include <vector>

using namespace machineLearning;

int main() {
    try {
        Dataset dataset =
            DataSetLoader::loadCSV("assets/User_Data.csv", 2);

        DatasetSplit split =
            DataSplitter::split(dataset, 0.8);

        Vector initialWeights(
            std::vector<double>(split.train.features.colCount(), 0.0)
        );

        LogisticModel initialModel(initialWeights, 0.0);

        Trainer trainer(0.1, 5000);

        LogisticRegressionPipeline pipeline(initialModel);

        LogisticTrainingResults results =
            pipeline.train(split.train, trainer);

        std::cout << "Final train loss: "
                  << results.losses.back() << "\n";

        Vector predictions =
            pipeline.predictClass(split.test.features);

        size_t correct = 0;

        for (size_t i = 0; i < predictions.size(); ++i) {
            int pred = static_cast<int>(predictions[i]);
            int target = static_cast<int>(split.test.targets[i]);

            if (pred == target) {
                ++correct;
            }
        }

        double accuracy =
            static_cast<double>(correct) / predictions.size();

        std::cout << "Test accuracy: "
                  << accuracy * 100.0 << "%\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

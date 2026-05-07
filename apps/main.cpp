#include "ml/data/DataSetLoader.h"
#include "ml/data/DataSplitter.h"
#include "ml/models/LogisticModel.h"
#include "ml/pipelines/LogisticRegressionPipeline.h"
#include "ml/training/Trainer.h"

#include <exception>
#include <iostream>
#include <vector>

using namespace machineLearning;

double squaredWeightNorm(const Vector& weights) {
    double sum = 0.0;

    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * weights[i];
    }

    return sum;
}

double accuracyScore(const Vector& predictions, const Vector& targets) {
    size_t correct = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred = static_cast<int>(predictions[i]);
        int target = static_cast<int>(targets[i]);

        if (pred == target) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / predictions.size();
}

void printResults(const std::string& label,
                  const LogisticTrainingResults& results,
                  double testAccuracy) {
    const LogisticModel& model = results.model;
    const Vector& weights = model.getWeights();

    std::cout << "\n=== " << label << " ===\n";

    std::cout << "Final data loss: "
              << results.dataLosses.back() << "\n";

    std::cout << "Final objective loss: "
              << results.trainingObjectiveLosses.back() << "\n";

    std::cout << "Test accuracy: "
              << testAccuracy * 100.0 << "%\n";

    std::cout << "Weights: ";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << weights[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Bias: "
              << model.getBias() << "\n";

    std::cout << "Squared weight norm: "
              << squaredWeightNorm(weights) << "\n";
}

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

        const double learningRate = 0.1;
        const size_t epochs = 5000;

        Trainer trainerNoReg(learningRate, epochs, 0.0);
        Trainer trainerReg(learningRate, epochs, 1.0);

        LogisticRegressionPipeline pipelineNoReg(initialModel);
        LogisticRegressionPipeline pipelineReg(initialModel);

        LogisticTrainingResults resultsNoReg =
            pipelineNoReg.train(split.train, trainerNoReg);

        LogisticTrainingResults resultsReg =
            pipelineReg.train(split.train, trainerReg);

        Vector predictionsNoReg =
            pipelineNoReg.predictClass(split.test.features);

        Vector predictionsReg =
            pipelineReg.predictClass(split.test.features);

        double accuracyNoReg =
            accuracyScore(predictionsNoReg, split.test.targets);

        double accuracyReg =
            accuracyScore(predictionsReg, split.test.targets);

        printResults("lambda = 0.0", resultsNoReg, accuracyNoReg);
        printResults("lambda = 1.0", resultsReg, accuracyReg);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
#include "ml/pipelines/LinearRegressionPipeline.h"
#include "ml/core/Matrix.h"
#include "ml/core/Vector.h"
#include "ml/training/Trainer.h"
#include "ml/training/LossFunctions.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace machineLearning;

namespace {

bool nearlyEqual(double a, double b, double eps = 1e-3) {
    return std::fabs(a - b) < eps;
}

void testPipelineTrainsAndPredicts() {
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

        targetData.push_back(3.0 * x1 + 5.0 * x2 + 2.0);
    }

    Matrix X(rows, cols, std::move(data));
    Vector y(std::move(targetData));

    LinearModel initialModel(Vector({0.1, 0.1}), 0.0);
    Trainer trainer(0.1, 1000);

    LinearRegressionPipeline pipeline(initialModel);

    TrainingResults result = pipeline.train(X, y, trainer);

    assert(result.trainingObjectiveLosses.front() > result.trainingObjectiveLosses.back());
    assert(result.trainingObjectiveLosses.back() < 1e-6);

    Vector predictions = pipeline.predict(X);

    double loss = meanSquaredError(predictions, y);
    assert(loss < 1e-6);

    assert(nearlyEqual(predictions[0], y[0]));
    assert(nearlyEqual(predictions[100], y[100]));
    assert(nearlyEqual(predictions[999], y[999]));
}

void testPredictBeforeTrainThrows() {
    LinearModel initialModel(Vector({0.1, 0.1}), 0.0);
    LinearRegressionPipeline pipeline(initialModel);

    Matrix X(1, 2, {1.0, 2.0});

    bool threw = false;

    try {
        pipeline.predict(X);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void testPipelinePredictsNewData() {
    const size_t rows = 500;
    const size_t cols = 2;

    std::vector<double> data;
    std::vector<double> targetData;

    data.reserve(rows * cols);
    targetData.reserve(rows);

    for (size_t i = 0; i < rows; ++i) {
        double x1 = static_cast<double>(i) / 500.0;
        double x2 = static_cast<double>(i % 20) / 10.0;

        data.push_back(x1);
        data.push_back(x2);

        targetData.push_back(3.0 * x1 + 5.0 * x2 + 2.0);
    }

    Matrix X(rows, cols, std::move(data));
    Vector y(std::move(targetData));

    LinearModel initialModel(Vector({0.1, 0.1}), 0.0);
    Trainer trainer(0.1, 1000);

    LinearRegressionPipeline pipeline(initialModel);
    pipeline.train(X, y, trainer);

    Matrix newX(3, 2, {
        0.25, 0.5,
        0.75, 1.2,
        1.5,  0.3
    });

    Vector predictions = pipeline.predict(newX);

    assert(nearlyEqual(predictions[0], 3.0 * 0.25 + 5.0 * 0.5 + 2.0));
    assert(nearlyEqual(predictions[1], 3.0 * 0.75 + 5.0 * 1.2 + 2.0));
    assert(nearlyEqual(predictions[2], 3.0 * 1.5  + 5.0 * 0.3 + 2.0));
}

}

int main() {
    testPipelineTrainsAndPredicts();
    testPredictBeforeTrainThrows();
    testPipelinePredictsNewData();

    std::cout << "All LinearRegressionPipeline tests passed.\n";
    return 0;
}

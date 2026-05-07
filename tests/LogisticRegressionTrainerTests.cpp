#include "ml/core/Vector.h"
#include "ml/core/Matrix.h"
#include "ml/models/LogisticModel.h"
#include "ml/training/LossFunctions.h"
#include "ml/training/Gradients.h"
#include "ml/training/Trainer.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace machineLearning;

bool nearlyEqual(double a, double b, double epsilon = 1e-6) {
    return std::fabs(a - b) < epsilon;
}

void testSigmoidProbabilityRange() {
    LogisticModel model(Vector({1.0, 1.0}), 0.0);

    Vector x({2.0, 3.0});
    double p = model.predictProbability(x);

    assert(p > 0.0);
    assert(p < 1.0);
}

void testPredictClass() {
    LogisticModel model(Vector({1.0, 1.0}), 0.0);

    Vector positiveX({2.0, 2.0});
    Vector negativeX({-2.0, -2.0});

    assert(model.predictClass(positiveX) == 1);
    assert(model.predictClass(negativeX) == 0);
}

void testMatrixProbabilityPrediction() {
    LogisticModel model(Vector({1.0, 1.0}), 0.0);

    Matrix X(
        3,
        2,
        {
            2.0, 2.0,
            0.0, 0.0,
            -2.0, -2.0
        }
    );

    Vector probabilities = model.predictProbability(X);

    assert(probabilities.size() == 3);
    assert(probabilities[0] > 0.5);
    assert(nearlyEqual(probabilities[1], 0.5));
    assert(probabilities[2] < 0.5);
}

void testBinaryCrossEntropyPerfectPredictionsLowLoss() {
    Vector probabilities({0.99, 0.01, 0.98, 0.02});
    Vector targets({1.0, 0.0, 1.0, 0.0});

    double loss = binaryCrossEntropy(probabilities, targets);

    assert(loss < 0.05);
}

void testBinaryCrossEntropyBadPredictionsHighLoss() {
    Vector probabilities({0.01, 0.99, 0.02, 0.98});
    Vector targets({1.0, 0.0, 1.0, 0.0});

    double loss = binaryCrossEntropy(probabilities, targets);

    assert(loss > 3.0);
}

void testBinaryCrossEntropyThrowsOnSizeMismatch() {
    Vector probabilities({0.5, 0.6});
    Vector targets({1.0});

    bool threw = false;

    try {
        binaryCrossEntropy(probabilities, targets);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void testBinaryCrossEntropyThrowsOnInvalidTarget() {
    Vector probabilities({0.5, 0.6});
    Vector targets({1.0, 0.5});

    bool threw = false;

    try {
        binaryCrossEntropy(probabilities, targets);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void testLogisticGradientsSimpleCase() {
    Matrix X(
        2,
        2,
        {
            1.0, 2.0,
            3.0, 4.0
        }
    );

    Vector probabilities({0.8, 0.4});
    Vector targets({1.0, 0.0});

    LogisticModelGradients gradients =
        computeLogisticModelBCEGradients(X, probabilities, targets);

    /*
        errors:
        sample 0: 0.8 - 1.0 = -0.2
        sample 1: 0.4 - 0.0 =  0.4

        grad w0 = average((-0.2 * 1) + (0.4 * 3))
                = average(-0.2 + 1.2)
                = 0.5

        grad w1 = average((-0.2 * 2) + (0.4 * 4))
                = average(-0.4 + 1.6)
                = 0.6

        grad b = average(-0.2 + 0.4)
               = 0.1
    */

    assert(nearlyEqual(gradients.weightGradients[0], 0.5));
    assert(nearlyEqual(gradients.weightGradients[1], 0.6));
    assert(nearlyEqual(gradients.biasGradient, 0.1));
}

void testLogisticTrainingLearnsSimpleDataset() {
    /*
        Simple separable dataset:

        Class 0:
        low x values

        Class 1:
        high x values

        The model should learn that larger x means class 1.
    */

    Matrix X(
        6,
        1,
        {
            0.0,
            1.0,
            2.0,
            8.0,
            9.0,
            10.0
        }
    );

    Vector y({0.0, 0.0, 0.0, 1.0, 1.0, 1.0});

    LogisticModel initialModel(Vector({0.0}), 0.0);

    Trainer trainer(0.1, 2000);

    LogisticTrainingResults results = trainer.train(initialModel, X, y);

    double firstLoss = results.losses.front();
    double finalLoss = results.losses.back();

    assert(finalLoss < firstLoss);

    assert(results.model.predictClass(Vector({1.0})) == 0);
    assert(results.model.predictClass(Vector({9.0})) == 1);

    double lowProbability = results.model.predictProbability(Vector({1.0}));
    double highProbability = results.model.predictProbability(Vector({9.0}));

    assert(lowProbability < 0.5);
    assert(highProbability > 0.5);
}

int main() {
    testSigmoidProbabilityRange();
    testPredictClass();
    testMatrixProbabilityPrediction();

    testBinaryCrossEntropyPerfectPredictionsLowLoss();
    testBinaryCrossEntropyBadPredictionsHighLoss();
    testBinaryCrossEntropyThrowsOnSizeMismatch();
    testBinaryCrossEntropyThrowsOnInvalidTarget();

    testLogisticGradientsSimpleCase();
    testLogisticTrainingLearnsSimpleDataset();

    std::cout << "All logistic regression tests passed.\n";

    return 0;
}

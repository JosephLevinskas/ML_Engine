#include "ml/models/LogisticModel.h"
#include "ml/core/Matrix.h"
#include "ml/core/Vector.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace machineLearning;

namespace {

bool nearlyEqual(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

void testPredictProbabilitySingleVector() {
    LogisticModel model(Vector({1.0, 1.0}), 0.0);

    Vector x({0.0, 0.0});

    double probability = model.predictProbability(x);

    assert(nearlyEqual(probability, 0.5));
}

void testPredictProbabilityPositiveScore() {
    LogisticModel model(Vector({2.0, 3.0}), 1.0);

    Vector x({1.0, 1.0});

    double probability = model.predictProbability(x);

    // z = 2*1 + 3*1 + 1 = 6
    double expected = 1.0 / (1.0 + std::exp(-6.0));

    assert(nearlyEqual(probability, expected));
    assert(probability > 0.5);
}

void testPredictProbabilityNegativeScore() {
    LogisticModel model(Vector({2.0, 3.0}), -10.0);

    Vector x({1.0, 1.0});

    double probability = model.predictProbability(x);

    // z = 2*1 + 3*1 - 10 = -5
    double expected = 1.0 / (1.0 + std::exp(5.0));

    assert(nearlyEqual(probability, expected));
    assert(probability < 0.5);
}

void testPredictProbabilityMatrix() {
    LogisticModel model(Vector({1.0, 1.0}), 0.0);

    Matrix X(3, 2, {
        0.0, 0.0,   // z = 0
        1.0, 1.0,   // z = 2
        -1.0, -1.0  // z = -2
    });

    Vector probabilities = model.predictProbability(X);

    assert(probabilities.size() == 3);

    assert(nearlyEqual(probabilities[0], 0.5));
    assert(nearlyEqual(probabilities[1], 1.0 / (1.0 + std::exp(-2.0))));
    assert(nearlyEqual(probabilities[2], 1.0 / (1.0 + std::exp(2.0))));
}

void testPredictClassSingleVector() {
    LogisticModel model(Vector({1.0, 1.0}), 0.0);

    Vector positive({1.0, 1.0});   // z = 2, prob > 0.5
    Vector negative({-1.0, -1.0}); // z = -2, prob < 0.5
    Vector boundary({0.0, 0.0});   // z = 0, prob = 0.5

    assert(nearlyEqual(model.predictClass(positive), 1.0));
    assert(nearlyEqual(model.predictClass(negative), 0.0));
    assert(nearlyEqual(model.predictClass(boundary), 1.0));
}

void testPredictClassMatrix() {
    LogisticModel model(Vector({1.0, 1.0}), 0.0);

    Matrix X(3, 2, {
        1.0, 1.0,    // class 1
        -1.0, -1.0,  // class 0
        0.0, 0.0     // boundary, class 1
    });

    Vector classes = model.predictClass(X);

    assert(classes.size() == 3);

    assert(nearlyEqual(classes[0], 1.0));
    assert(nearlyEqual(classes[1], 0.0));
    assert(nearlyEqual(classes[2], 1.0));
}

void testWrongVectorSizeThrows() {
    LogisticModel model(Vector({1.0, 1.0}), 0.0);

    Vector badX({1.0, 2.0, 3.0});

    bool threw = false;

    try {
        model.predictProbability(badX);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void testGetters() {
    LogisticModel model(Vector({3.0, 5.0}), 2.0);

    assert(nearlyEqual(model.getWeights()[0], 3.0));
    assert(nearlyEqual(model.getWeights()[1], 5.0));
    assert(nearlyEqual(model.getBias(), 2.0));
}

}

int main() {
    testPredictProbabilitySingleVector();
    testPredictProbabilityPositiveScore();
    testPredictProbabilityNegativeScore();
    testPredictProbabilityMatrix();
    testPredictClassSingleVector();
    testPredictClassMatrix();
    testWrongVectorSizeThrows();
    testGetters();

    std::cout << "All LogisticModel tests passed.\n";
    return 0;
}

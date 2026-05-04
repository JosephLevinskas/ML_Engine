#include "ml/StandardScaler.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace machineLearning {

bool nearlyEqual(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

// Helper: creates a scaler that can exist without needing Vector default constructor.
// This assumes your StandardScaler has a constructor like:
// StandardScaler(Vector means_, Vector stdDevs_, bool fitted_)
StandardScaler makeUnfittedScaler() {
    return StandardScaler(
        Vector({0.0}),
        Vector({1.0}),
        false
    );
}

void testFitTransformScalesColumns() {
    Matrix X(3, 2, {
        1.0, 10.0,
        2.0, 20.0,
        3.0, 30.0
    });

    StandardScaler scaler = makeUnfittedScaler();
    Matrix scaled = scaler.fitTransform(X);

    const double s = std::sqrt(2.0 / 3.0);

    assert(nearlyEqual(scaled(0, 0), (1.0 - 2.0) / s));
    assert(nearlyEqual(scaled(1, 0), 0.0));
    assert(nearlyEqual(scaled(2, 0), (3.0 - 2.0) / s));

    assert(nearlyEqual(scaled(0, 1), (10.0 - 20.0) / (10.0 * s)));
    assert(nearlyEqual(scaled(1, 1), 0.0));
    assert(nearlyEqual(scaled(2, 1), (30.0 - 20.0) / (10.0 * s)));
}

void testTransformDoesNotMutateInput() {
    Matrix X(2, 2, {
        1.0, 100.0,
        3.0, 200.0
    });

    StandardScaler scaler = makeUnfittedScaler();
    Matrix scaled = scaler.fitTransform(X);

    assert(nearlyEqual(X(0, 0), 1.0));
    assert(nearlyEqual(X(0, 1), 100.0));
    assert(nearlyEqual(X(1, 0), 3.0));
    assert(nearlyEqual(X(1, 1), 200.0));

    assert(!nearlyEqual(scaled(0, 0), X(0, 0)));
}

void testConstantColumnDoesNotDivideByZero() {
    Matrix X(3, 2, {
        5.0, 1.0,
        5.0, 2.0,
        5.0, 3.0
    });

    StandardScaler scaler = makeUnfittedScaler();
    Matrix scaled = scaler.fitTransform(X);

    assert(nearlyEqual(scaled(0, 0), 0.0));
    assert(nearlyEqual(scaled(1, 0), 0.0));
    assert(nearlyEqual(scaled(2, 0), 0.0));
}

void testTransformBeforeFitThrows() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    StandardScaler scaler = makeUnfittedScaler();

    bool threw = false;

    try {
        scaler.transform(X);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void testTransformWrongColumnCountThrows() {
    Matrix trainX(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Matrix badX(2, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    });

    StandardScaler scaler = makeUnfittedScaler();
    scaler.fit(trainX);

    bool threw = false;

    try {
        scaler.transform(badX);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void testSeparateFitThenTransform() {
    Matrix trainX(3, 1, {
        10.0,
        20.0,
        30.0
    });

    Matrix newX(2, 1, {
        20.0,
        40.0
    });

    StandardScaler scaler = makeUnfittedScaler();
    scaler.fit(trainX);

    Matrix scaled = scaler.transform(newX);

    const double mean = 20.0;
    const double stdDev = std::sqrt(200.0 / 3.0);

    assert(nearlyEqual(scaled(0, 0), (20.0 - mean) / stdDev));
    assert(nearlyEqual(scaled(1, 0), (40.0 - mean) / stdDev));
}

}

int main() {
    machineLearning::testFitTransformScalesColumns();
    machineLearning::testTransformDoesNotMutateInput();
    machineLearning::testConstantColumnDoesNotDivideByZero();
    machineLearning::testTransformBeforeFitThrows();
    machineLearning::testTransformWrongColumnCountThrows();
    machineLearning::testSeparateFitThenTransform();

    std::cout << "All StandardScaler tests passed.\n";
    return 0;
}
#include "ml/models/LinearModel.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

using namespace machineLearning;

void test_single_prediction_valid() {
    LinearModel model(Vector({2.0, 3.0}), 5.0);

    Vector x({10.0, 20.0});

    double prediction = model.predict(x);

    // (10 * 2) + (20 * 3) + 5 = 85
    assert(prediction == 85.0);
}

void test_batch_prediction_valid() {
    LinearModel model(Vector({2.0, 3.0}), 5.0);

    Matrix X(3, 2, {
        10.0, 20.0,
        1.0,  2.0,
        0.0,  4.0
    });

    Vector predictions = model.predict(X);

    assert(predictions.size() == 3);

    // row 0: 10*2 + 20*3 + 5 = 85
    assert(predictions[0] == 85.0);

    // row 1: 1*2 + 2*3 + 5 = 13
    assert(predictions[1] == 13.0);

    // row 2: 0*2 + 4*3 + 5 = 17
    assert(predictions[2] == 17.0);
}

void test_single_prediction_dimension_mismatch_throws() {
    LinearModel model(Vector({2.0, 3.0}), 5.0);

    Vector x({1.0, 2.0, 3.0});

    bool threw = false;

    try {
        model.predict(x);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_batch_prediction_dimension_mismatch_throws() {
    LinearModel model(Vector({2.0, 3.0}), 5.0);

    Matrix X(2, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    });

    bool threw = false;

    try {
        model.predict(X);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_zero_bias_prediction() {
    LinearModel model(Vector({1.0, -1.0}), 0.0);

    Vector x({10.0, 4.0});

    double prediction = model.predict(x);

    // 10*1 + 4*(-1) + 0 = 6
    assert(prediction == 6.0);
}

void test_negative_bias_prediction() {
    LinearModel model(Vector({4.0, 2.0}), -10.0);

    Vector x({3.0, 5.0});

    double prediction = model.predict(x);

    // 3*4 + 5*2 - 10 = 12
    assert(prediction == 12.0);
}

int main() {
    test_single_prediction_valid();
    test_batch_prediction_valid();
    test_single_prediction_dimension_mismatch_throws();
    test_batch_prediction_dimension_mismatch_throws();
    test_zero_bias_prediction();
    test_negative_bias_prediction();

    std::cout << "All LinearModel tests passed.\n";
    return 0;
}

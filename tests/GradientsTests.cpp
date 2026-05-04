#include "ml/Gradients.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

using namespace machineLearning;

void test_linear_model_mse_gradients_simple_case() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Vector predictions({10.0, 20.0});
    Vector targets({8.0, 15.0});

    LinearModelGradients gradients =
        computeLinearModelMSEGradients(X, predictions, targets);

    // errors = [2, 5]
    // n = 2
    //
    // weight gradient 0:
    // (2/n) * (2*1 + 5*3) = 1 * 17 = 17
    //
    // weight gradient 1:
    // (2/n) * (2*2 + 5*4) = 1 * 24 = 24
    //
    // bias gradient:
    // (2/n) * (2 + 5) = 1 * 7 = 7

    assert(gradients.weightGradients.size() == 2);
    assert(gradients.weightGradients[0] == 17.0);
    assert(gradients.weightGradients[1] == 24.0);
    assert(gradients.biasGradient == 7.0);
}

void test_linear_model_mse_gradients_zero_error() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Vector predictions({5.0, 9.0});
    Vector targets({5.0, 9.0});

    LinearModelGradients gradients =
        computeLinearModelMSEGradients(X, predictions, targets);

    assert(gradients.weightGradients.size() == 2);
    assert(gradients.weightGradients[0] == 0.0);
    assert(gradients.weightGradients[1] == 0.0);
    assert(gradients.biasGradient == 0.0);
}

void test_linear_model_mse_gradients_negative_error() {
    Matrix X(1, 2, {
        2.0, 3.0
    });

    Vector predictions({4.0});
    Vector targets({10.0});

    LinearModelGradients gradients =
        computeLinearModelMSEGradients(X, predictions, targets);

    // error = -6
    // n = 1
    //
    // weight gradient 0 = 2 * -6 * 2 = -24
    // weight gradient 1 = 2 * -6 * 3 = -36
    // bias gradient = 2 * -6 = -12

    assert(gradients.weightGradients.size() == 2);
    assert(gradients.weightGradients[0] == -24.0);
    assert(gradients.weightGradients[1] == -36.0);
    assert(gradients.biasGradient == -12.0);
}

void test_predictions_size_mismatch_throws() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Vector predictions({1.0});
    Vector targets({1.0, 2.0});

    bool threw = false;

    try {
        computeLinearModelMSEGradients(X, predictions, targets);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_targets_size_mismatch_throws() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Vector predictions({1.0, 2.0});
    Vector targets({1.0});

    bool threw = false;

    try {
        computeLinearModelMSEGradients(X, predictions, targets);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

int main() {
    test_linear_model_mse_gradients_simple_case();
    test_linear_model_mse_gradients_zero_error();
    test_linear_model_mse_gradients_negative_error();
    test_predictions_size_mismatch_throws();
    test_targets_size_mismatch_throws();

    std::cout << "All Gradients tests passed.\n";
    return 0;
}
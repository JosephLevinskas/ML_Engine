#include "ml/training/LossFunctions.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

using namespace machineLearning;

void test_mean_squared_error_zero_error() {
    Vector predictions({1.0, 2.0, 3.0});
    Vector targets({1.0, 2.0, 3.0});

    assert(meanSquaredError(predictions, targets) == 0.0);
}

void test_mean_squared_error_valid() {
    Vector predictions({2.0, 6.0, 8.0});
    Vector targets({3.0, 5.0, 7.0});

    // errors = [-1, 1, 1]
    // squared = [1, 1, 1]
    // mean = 1
    assert(meanSquaredError(predictions, targets) == 1.0);
}

void test_mean_squared_error_larger_values() {
    Vector predictions({10.0, 20.0});
    Vector targets({4.0, 8.0});

    // errors = [6, 12]
    // squared = [36, 144]
    // mean = 90
    assert(meanSquaredError(predictions, targets) == 90.0);
}

void test_mean_squared_error_negative_errors() {
    Vector predictions({1.0, 2.0});
    Vector targets({4.0, 6.0});

    // errors = [-3, -4]
    // squared = [9, 16]
    // mean = 12.5
    assert(meanSquaredError(predictions, targets) == 12.5);
}

void test_mean_squared_error_size_mismatch_throws() {
    Vector predictions({1.0, 2.0});
    Vector targets({1.0, 2.0, 3.0});

    bool threw = false;

    try {
        meanSquaredError(predictions, targets);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

int main() {
    test_mean_squared_error_zero_error();
    test_mean_squared_error_valid();
    test_mean_squared_error_larger_values();
    test_mean_squared_error_negative_errors();
    test_mean_squared_error_size_mismatch_throws();

    std::cout << "All LossFunctions tests passed.\n";
    return 0;
}

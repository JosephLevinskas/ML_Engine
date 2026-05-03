#include "ml/Operations.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

using namespace machineLearning;

void test_dot_product_valid() {
    Vector a({1.0, 2.0, 3.0});
    Vector b({10.0, 20.0, 30.0});

    assert(dot(a, b) == 140.0);
}

void test_dot_product_size_mismatch_throws() {
    Vector a({1.0, 2.0});
    Vector b({1.0, 2.0, 3.0});

    bool threw = false;

    try {
        dot(a, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_matrix_vector_multiply_valid() {
    Matrix m(2, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    });

    Vector v({10.0, 20.0, 30.0});

    Vector result = m * v;

    assert(result.size() == 2);
    assert(result[0] == 140.0);
    assert(result[1] == 320.0);
}

void test_matrix_vector_multiply_dimension_mismatch_throws() {
    Matrix m(2, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    });

    Vector v({1.0, 2.0});

    bool threw = false;

    try {
        Vector result = m * v;
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_matrix_vector_multiply_zero_vector() {
    Matrix m(2, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    });

    Vector v({0.0, 0.0, 0.0});

    Vector result = m * v;

    assert(result.size() == 2);
    assert(result[0] == 0.0);
    assert(result[1] == 0.0);
}

void test_matrix_vector_multiply_identity_like_matrix() {
    Matrix m(3, 3, {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    });

    Vector v({7.0, -2.0, 5.0});

    Vector result = m * v;

    assert(result.size() == 3);
    assert(result[0] == 7.0);
    assert(result[1] == -2.0);
    assert(result[2] == 5.0);
}

int main() {
    test_dot_product_valid();
    test_dot_product_size_mismatch_throws();
    test_matrix_vector_multiply_valid();
    test_matrix_vector_multiply_dimension_mismatch_throws();
    test_matrix_vector_multiply_zero_vector();
    test_matrix_vector_multiply_identity_like_matrix();

    std::cout << "All Operations tests passed.\n";
    return 0;
}
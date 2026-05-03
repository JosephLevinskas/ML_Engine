#include "ml/Vector.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

using namespace machineLearning;

void test_constructor_valid() {
    Vector v({1.0, 2.0, 3.0});

    assert(v.size() == 3);
    assert(v[0] == 1.0);
    assert(v[1] == 2.0);
    assert(v[2] == 3.0);
}

void test_constructor_empty_throws() {
    bool threw = false;

    try {
        Vector v({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_const_indexing() {
    const Vector v({4.0, 5.0, 6.0});

    assert(v[0] == 4.0);
    assert(v[1] == 5.0);
    assert(v[2] == 6.0);
}

void test_mutable_indexing() {
    Vector v({1.0, 2.0, 3.0});

    v[1] = 10.0;

    assert(v[0] == 1.0);
    assert(v[1] == 10.0);
    assert(v[2] == 3.0);
}

void test_index_out_of_range_throws() {
    Vector v({1.0, 2.0, 3.0});

    bool threw = false;

    try {
        v[3];
    } catch (const std::out_of_range&) {
        threw = true;
    }

    assert(threw);
}

void test_add_assign_valid() {
    Vector a({1.0, 2.0, 3.0});
    Vector b({4.0, 5.0, 6.0});

    a += b;

    assert(a[0] == 5.0);
    assert(a[1] == 7.0);
    assert(a[2] == 9.0);
}

void test_add_assign_size_mismatch_throws() {
    Vector a({1.0, 2.0});
    Vector b({1.0, 2.0, 3.0});

    bool threw = false;

    try {
        a += b;
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_subtract_assign_valid() {
    Vector a({10.0, 20.0, 30.0});
    Vector b({1.0, 2.0, 3.0});

    a -= b;

    assert(a[0] == 9.0);
    assert(a[1] == 18.0);
    assert(a[2] == 27.0);
}

void test_subtract_assign_size_mismatch_throws() {
    Vector a({1.0, 2.0});
    Vector b({1.0, 2.0, 3.0});

    bool threw = false;

    try {
        a -= b;
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_scalar_multiply_assign() {
    Vector v({1.0, -2.0, 3.0});

    v *= 3.0;

    assert(v[0] == 3.0);
    assert(v[1] == -6.0);
    assert(v[2] == 9.0);
}

void test_add_operator() {
    Vector a({1.0, 2.0, 3.0});
    Vector b({4.0, 5.0, 6.0});

    Vector result = a + b;

    assert(result[0] == 5.0);
    assert(result[1] == 7.0);
    assert(result[2] == 9.0);
}

void test_subtract_operator() {
    Vector a({10.0, 20.0, 30.0});
    Vector b({1.0, 2.0, 3.0});

    Vector result = a - b;

    assert(result[0] == 9.0);
    assert(result[1] == 18.0);
    assert(result[2] == 27.0);
}

void test_vector_scalar_multiply() {
    Vector v({1.0, -2.0, 3.0});

    Vector result = v * 2.0;

    assert(result[0] == 2.0);
    assert(result[1] == -4.0);
    assert(result[2] == 6.0);
}

void test_scalar_vector_multiply() {
    Vector v({1.0, -2.0, 3.0});

    Vector result = 2.0 * v;

    assert(result[0] == 2.0);
    assert(result[1] == -4.0);
    assert(result[2] == 6.0);
}

void test_non_mutating_operators_do_not_change_inputs() {
    Vector a({1.0, 2.0, 3.0});
    Vector b({4.0, 5.0, 6.0});

    Vector sum = a + b;
    Vector diff = a - b;
    Vector scaled = a * 10.0;

    assert(a[0] == 1.0);
    assert(a[1] == 2.0);
    assert(a[2] == 3.0);

    assert(b[0] == 4.0);
    assert(b[1] == 5.0);
    assert(b[2] == 6.0);
}

int main() {
    test_constructor_valid();
    test_constructor_empty_throws();
    test_const_indexing();
    test_mutable_indexing();
    test_index_out_of_range_throws();

    test_add_assign_valid();
    test_add_assign_size_mismatch_throws();
    test_subtract_assign_valid();
    test_subtract_assign_size_mismatch_throws();

    test_scalar_multiply_assign();
    test_add_operator();
    test_subtract_operator();
    test_vector_scalar_multiply();
    test_scalar_vector_multiply();
    test_non_mutating_operators_do_not_change_inputs();

    std::cout << "All Vector tests passed.\n";
    return 0;
}
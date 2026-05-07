#include "ml/core/Matrix.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace machineLearning;

bool almostEqual(double a, double b) {
    return std::abs(a - b) < 1e-9;
}

int main() {
    {
        Matrix m(2, 3);

        assert(m.rowCount() == 2);
        assert(m.colCount() == 3);

        assert(almostEqual(m(0, 0), 0.0));
        assert(almostEqual(m(1, 2), 0.0));
    }

    {
        Matrix m(2, 3, {1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0});

        assert(m.rowCount() == 2);
        assert(m.colCount() == 3);

        assert(almostEqual(m(0, 0), 1.0));
        assert(almostEqual(m(0, 2), 3.0));
        assert(almostEqual(m(1, 0), 4.0));
        assert(almostEqual(m(1, 2), 6.0));
    }

    {
        Matrix m(2, 2);

        m(0, 0) = 10.0;
        m(1, 1) = 20.0;

        assert(almostEqual(m(0, 0), 10.0));
        assert(almostEqual(m(1, 1), 20.0));
    }

    {
        bool threw = false;
        try {
            Matrix m(0, 3);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    {
        bool threw = false;
        try {
            Matrix m(3, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    {
        bool threw = false;
        try {
            Matrix m(2, 3, {1.0, 2.0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    {
        Matrix m(2, 3);

        bool rowThrew = false;
        try {
            m(2, 0);
        } catch (const std::out_of_range&) {
            rowThrew = true;
        }
        assert(rowThrew);

        bool colThrew = false;
        try {
            m(0, 3);
        } catch (const std::out_of_range&) {
            colThrew = true;
        }
        assert(colThrew);
    }

    return 0;
}

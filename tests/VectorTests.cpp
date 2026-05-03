#include "ml/Vector.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace machineLearning;

bool almostEqual(double a, double b) {
    return std::abs(a - b) < 1e-9;
}

int main() {
    {
        Vector v({1.0, 2.0, 3.0});
        assert(v.size() == 3);
        assert(almostEqual(v[0], 1.0));
        assert(almostEqual(v[2], 3.0));
    }

    {
        bool threw = false;
        try {
            Vector v({});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    {
        Vector v({1.0, 2.0, 3.0});
        v[1] = 10.0;
        assert(almostEqual(v[1], 10.0));
    }

    {
        Vector v({1.0, 2.0});
        bool threw = false;
        try {
            v[2];
        } catch (const std::out_of_range&) {
            threw = true;
        }
        assert(threw);
    }

    {
        Vector a({1.0, 2.0, 3.0});
        Vector b({4.0, 5.0, 6.0});
        assert(almostEqual(dot(a, b), 32.0));
    }

    {
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

    {
        Vector a({1.0, 2.0, 3.0});
        Vector b({4.0, 5.0, 6.0});

        Vector c = a + b;
        assert(almostEqual(c[0], 5.0));
        assert(almostEqual(c[1], 7.0));
        assert(almostEqual(c[2], 9.0));
    }

    {
        Vector a({5.0, 7.0, 9.0});
        Vector b({4.0, 5.0, 6.0});

        Vector c = a - b;
        assert(almostEqual(c[0], 1.0));
        assert(almostEqual(c[1], 2.0));
        assert(almostEqual(c[2], 3.0));
    }

    {
        Vector v({1.0, 2.0, 3.0});

        Vector a = v * 2.0;
        Vector b = 2.0 * v;

        assert(almostEqual(a[0], 2.0));
        assert(almostEqual(a[1], 4.0));
        assert(almostEqual(a[2], 6.0));

        assert(almostEqual(b[0], 2.0));
        assert(almostEqual(b[1], 4.0));
        assert(almostEqual(b[2], 6.0));
    }

    {
        Vector a({1.0, 2.0});
        Vector b({3.0, 4.0});

        a += b;
        assert(almostEqual(a[0], 4.0));
        assert(almostEqual(a[1], 6.0));

        a -= b;
        assert(almostEqual(a[0], 1.0));
        assert(almostEqual(a[1], 2.0));

        a *= 3.0;
        assert(almostEqual(a[0], 3.0));
        assert(almostEqual(a[1], 6.0));
    }

    {
        Vector a({1.0, 2.0});
        Vector b({1.0, 2.0, 3.0});

        bool addThrew = false;
        try {
            a + b;
        } catch (const std::invalid_argument&) {
            addThrew = true;
        }
        assert(addThrew);

        bool subThrew = false;
        try {
            a - b;
        } catch (const std::invalid_argument&) {
            subThrew = true;
        }
        assert(subThrew);
    }

    return 0;
}
#include "ml/vector.h"

#include <stdexcept>
#include <utility>

namespace machineLearning {

Vector::Vector(std::vector<double> data_)
    : data(std::move(data_))
{
    if (data.empty()) {
        throw std::invalid_argument("Cannot create empty vector");
    }
}

size_t Vector::size() const {
    return data.size();
}

double Vector::operator[](size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Vector index out of bounds");
    }
    return data[index];
}

double& Vector::operator[](size_t index) {
    if (index >= data.size()) {
        throw std::out_of_range("Vector index out of bounds");
    }
    return data[index];
}

double dot(const Vector& v1, const Vector& v2) {
    const size_t size = v1.size();

    if (size != v2.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }

    double sum = 0.0;

    for (size_t i = 0; i < size; ++i) {
        sum += v1[i] * v2[i];
    }

    return sum;

}

Vector& Vector::operator+=(const Vector& other) {
    size_t size = data.size();
    if (size != other.size()) {
        throw std::invalid_argument("Vector sizes must match for +=");
    }

    for (size_t i = 0; i < size; ++i) {
        data[i] += other[i];
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    size_t size = data.size();
    if (size != other.size()) {
        throw std::invalid_argument("Vector sizes must match for -=");
    }

    for (size_t i = 0; i < size; ++i) {
        data[i] -= other[i];
    }
    return *this;
}

Vector& Vector::operator*=(double scalar) {
    size_t size = data.size();
    for (size_t i = 0; i < size; ++i) {
        data[i] *= scalar;
    }
    return *this;
}

Vector operator+(const Vector& a, const Vector& b) {
    Vector v = a;
    return v += b;
}

Vector operator-(const Vector& a, const Vector& b) {
    Vector v = a;
    return v -= b;
}

Vector operator*(const Vector& v, double scalar) {
    Vector newV = v;
    return newV *= scalar;
}

Vector operator*(double scalar, const Vector& v) {
    return v * scalar;
}
}
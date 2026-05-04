#pragma once

#include <vector>
#include <cstddef>


namespace machineLearning {

class Vector {
private:
    std::vector<double> data;

public:
    explicit Vector(std::vector<double> data_);

    size_t size() const;

    double operator[](size_t index) const;
    double& operator[](size_t index);

    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(double scalar);
    Vector& operator+=(double scalar);
    Vector& operator-=(double scalar);

    

};




Vector operator+(const Vector& a, const Vector& b);
Vector operator-(const Vector& a, const Vector& b);
Vector operator*(const Vector& v, double scalar);
Vector operator*(double scalar, const Vector& v);
Vector operator+(const Vector& v, double scalar);
Vector operator+(double scalar, const Vector& v);
}